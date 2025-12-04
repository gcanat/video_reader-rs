# B-Frame 视频 Seek 偏移问题修复记录

## 问题背景

在处理包含 B-Frame 的视频时，`get_batch(indices, with_fallback=False)` 返回的帧与迭代器 (`for frame in vr`) 返回的帧不一致，存在固定的偏移（如 +7 或 -7）。

### 问题视频特征
- 包含 B-Frame（双向预测帧）
- 视频开头有负 PTS（Presentation Timestamp）的帧
- 解码顺序 ≠ 显示顺序

## 根本原因分析

### 1. 负 PTS 帧导致的偏移

视频开头的帧有负 PTS：
```
帧 0: PTS = -0.25s  (负值)
帧 1: PTS = -0.18s  (负值)
...
帧 6: PTS = 0.0s
帧 7: PTS = -0.017s (还是负值！)
```

总共有 7 帧的 PTS < 0，这会导致 seek 后的帧索引偏移。

### 2. 从头遍历 vs Seek 的行为差异

**从头遍历（iterator）：**
```
┌─────────────────────────────────────────────────────┐
│ 解码: packet 0 → packet 1 → packet 2 → ...          │
│ 输出: frame 0, frame 1, frame 2, ... （顺序计数）    │
│ 所有负 PTS 帧都被正确计入                            │
└─────────────────────────────────────────────────────┘
```

**Seek 到 keyframe 56：**
```
┌─────────────────────────────────────────────────────┐
│ 跳过 packet 0-55，从 packet 56 开始解码              │
│ 解码器"重置"，不知道之前有 7 个负 PTS 帧             │
│ 导致帧索引偏移 +7                                    │
└─────────────────────────────────────────────────────┘
```

### 3. 跨 GOP 时 `can_skip_forward` 的错误判断

当索引列表包含跨 GOP 的帧时（如 `[55, 58]`）：

```rust
// 处理 frame 55 后：
curr_dec_idx = 56  // 在"无偏移空间"

// 处理 frame 58：
adjusted_index = 58 + 7 = 65  // 有偏移
keyframe = 56
can_skip_forward = (65 >= 56) && (56 >= 56) = True  // ❌ 错误！

// 直接从 56 skip 到 65，但此时 curr_dec_idx 的含义已变！
// 结果：返回 frame 65，而不是 frame 58
```

## 修复方案

### 修复 1：计算并存储负 PTS 偏移量 (`info.rs`)

```rust
pub struct StreamInfo {
    // ... 原有字段 ...
    
    /// 负 PTS 帧数量，用于 seek 偏移校正
    negative_pts_offset: usize,
}

impl StreamInfo {
    pub fn new(...) -> Self {
        // 计算负 PTS 帧数量
        let negative_pts_offset = frame_times
            .values()
            .filter(|ft| ft.pts < 0)
            .count();
        // ...
    }
    
    pub fn negative_pts_offset(&self) -> usize {
        self.negative_pts_offset
    }
}
```

### 修复 2：条件性应用偏移量 (`reader.rs`)

**核心逻辑：只在 seek 到非首个 keyframe 时应用偏移**

```rust
pub fn seek_accurate_raw(&mut self, frame_index: usize) -> Result<Option<Video>, ffmpeg::Error> {
    // 1. 先找原始 keyframe
    let key_decode_idx_orig = self.locate_keyframes(&frame_index);
    
    // 2. 只有 keyframe > 0 时才应用偏移
    //    对于 keyframe=0（第一个 GOP），从头解码，无需偏移
    let offset = if key_decode_idx_orig > 0 {
        self.stream_info.negative_pts_offset()
    } else {
        0
    };
    let adjusted_index = frame_index.saturating_add(offset);
    
    // 3. 用调整后的索引重新找 keyframe
    let key_decode_idx = if offset > 0 {
        self.locate_keyframes(&adjusted_index)
    } else {
        key_decode_idx_orig
    };
    
    // ... 后续处理 ...
}
```

### 修复 3：跨 GOP 时强制 seek (`reader.rs`)

**关键修复：当 offset > 0 时，不允许直接 skip_forward**

```rust
// 检查能否直接 skip forward
let can_skip_forward = if offset > 0 {
    // 有偏移时，只有当 curr_dec_idx 已经超过了原始 keyframe 才能 skip
    // 这意味着之前已经 seek 到这个 keyframe 了
    (adjusted_index >= self.curr_dec_idx)
        && (self.curr_dec_idx >= key_decode_idx)
        && (self.curr_dec_idx > key_decode_idx_orig)  // ✅ 新增条件！
} else {
    // 无偏移时，正常逻辑
    (adjusted_index >= self.curr_dec_idx)
        && (self.curr_dec_idx >= key_decode_idx)
};

if can_skip_forward {
    // 直接 skip
} else {
    // 必须 seek
    if key_decode_idx < self.curr_dec_idx || offset > 0 {
        self.seek_to_start()?;  // 先回到开头
    }
    self.seek_frame_by_decode_idx(&key_decode_idx)?;
    // ...
}
```

### 修复 4：完整刷新解码器和滤镜缓冲区 (`reader.rs`)

```rust
pub fn avflushbuf(&mut self) -> Result<(), ffmpeg::Error> {
    // 刷新解码器缓冲区
    unsafe { avcodec_flush_buffers(self.decoder.video.as_mut_ptr()) };
    
    // 清空解码器中残留的帧
    let mut decoded = Video::empty();
    while self.decoder.video.receive_frame(&mut decoded).is_ok() {
        debug!("Discarding buffered decoder frame after flush");
    }
    
    // 清空滤镜图中残留的帧（关键！）
    let mut filter_frame = Video::empty();
    while self.decoder.graph.get("out").unwrap().sink().frame(&mut filter_frame).is_ok() {
        debug!("Discarding buffered filter frame after flush");
    }
    
    Ok(())
}
```

## 其他优化

### 1. 异步 YUV→RGB 转换 (`get_batch` 和 `get_batch_safe`)

```rust
// 收集 YUV 帧，异步转换
let mut tasks: Vec<(usize, thread::JoinHandle<FrameArray>)> = Vec::new();

for frame_index in unique_frames {
    if let Some(yuv_frame) = self.seek_accurate_raw(frame_index)? {
        tasks.push((
            frame_index,
            thread::spawn(move || {
                convert_yuv_to_ndarray_rgb24(yuv_frame, cspace, crange)
            }),
        ));
    }
}

// 收集结果
for (frame_idx, task) in tasks {
    let frame = task.join().unwrap();
    frame_map.insert(frame_idx, frame);
}
```

### 2. 索引排序和去重

```rust
pub fn get_batch(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
    // 排序以最小化回退 seek
    let mut sorted_indices: Vec<(usize, usize)> = indices.iter()
        .enumerate()
        .map(|(orig_idx, &frame_idx)| (orig_idx, frame_idx))
        .collect();
    sorted_indices.sort_by_key(|&(_, frame_idx)| frame_idx);
    
    // 去重避免重复解码
    let unique_frames: Vec<usize> = {
        let mut seen = HashSet::new();
        sorted_indices.iter()
            .filter_map(|&(_, frame_idx)| {
                if seen.insert(frame_idx) { Some(frame_idx) } else { None }
            })
            .collect()
    };
    // ...
}
```

### 3. 二分查找定位 keyframe

```rust
pub fn locate_keyframes(&self, pos: &usize) -> usize {
    let key_frames = self.stream_info.key_frames();
    if key_frames.is_empty() {
        return 0;
    }
    
    // O(log K) 代替 O(K)
    match key_frames.binary_search(pos) {
        Ok(idx) => key_frames[idx],
        Err(idx) => {
            if idx == 0 { 0 } else { key_frames[idx - 1] }
        }
    }
}
```

### 4. 自动选择最快方法 (`with_fallback=None`)

```rust
pub fn estimate_decode_cost(&self, indices: &[usize]) -> (usize, usize) {
    // seek_cost: 每个帧从其 keyframe 解码的总帧数
    // sequential_cost: max_index + 1
    // ...
}

pub fn should_use_sequential(&self, indices: &[usize]) -> bool {
    let (seek_cost, sequential_cost) = self.estimate_decode_cost(indices);
    let seek_overhead = 1.5;  // seek I/O 开销因子
    (seek_cost as f64 * seek_overhead) > sequential_cost as f64
}
```

## 测试验证

```python
import numpy as np
from video_reader import PyVideoReader

vid_path = 'your_video.mov'

# 获取 ground truth
vr = PyVideoReader(vid_path)
ground_truth = np.array([frame for frame in vr])

# 测试各种索引组合
batch_indexs = [55, 58, 60, 100, 108, 115]  # 跨 GOP

vr2 = PyVideoReader(vid_path)
batch = vr2.get_batch(batch_indexs, with_fallback=False)

# 验证
for i, idx in enumerate(batch_indexs):
    eq = np.array_equal(batch[i], ground_truth[idx])
    print(f'idx {idx}: {"OK" if eq else "FAIL"}')
```

## 总结

| 问题 | 原因 | 修复 |
|------|------|------|
| 固定偏移 (+7/-7) | 负 PTS 帧导致 seek 后索引错位 | 计算 `negative_pts_offset` 并条件性应用 |
| 第一个 GOP 正确，后续 GOP 错误 | 首个 GOP 从头解码无偏移，后续 GOP seek 有偏移 | 只在 `keyframe > 0` 时应用偏移 |
| 跨 GOP 时错误 | `can_skip_forward` 错误判断 | 增加 `curr_dec_idx > key_decode_idx_orig` 条件 |
| seek 后残留帧 | 滤镜图未刷新 | `avflushbuf` 同时清空解码器和滤镜缓冲区 |

## 适用范围

此修复适用于所有满足以下条件的视频：
1. 包含 B-Frame（双向预测帧）
2. 视频开头有负 PTS 帧
3. 使用 seek-based 方法 (`with_fallback=False`) 获取帧

对于无 B-Frame 或无负 PTS 的视频，`negative_pts_offset = 0`，不会影响原有逻辑。

---

# 2024-12 新增修复：FFmpeg Seek 完全失效的情况

## 新发现：负 PTS/DTS 视频的 Seek 根本不可靠

经过深入测试，发现对于某些视频，FFmpeg 的 `av_seek_frame` 会 **完全失效**，seek 到错误的位置。

### 问题视频类型

| 类型 | 示例 | 问题 |
|------|------|------|
| **负 PTS** | `ea09afcd-*.mov` (pts=-150, -110, ...) | seek 到 keyframe 56 返回开头帧 |
| **负 DTS** | `out_C_negpts.mp4` (pts=0, dts=-66667) | seek 到 keyframe 493 返回 keyframe 249 |

### 实验验证

**负 PTS 视频：**
```
seek to pts=489 (keyframe 56)
    ↓
期望: 解码 keyframe 56 的 GOP
实际: 解码到 GT[7] (视频开头 pts=0 的帧)
```

**负 DTS 视频：**
```
seek to pts=16432347 (keyframe 493)
    ↓
期望: 第一帧 pts=16432347
实际: 第一帧 pts=8300000 (keyframe 249 的位置！)
```

### 根本原因

FFmpeg 在处理负 PTS/DTS 时会进行内部时间戳归一化，导致：
1. `av_seek_frame()` 的目标位置被错误计算
2. Seek 可能跳到完全不相关的 keyframe
3. 解码器输出的帧 PTS 与 seek 目标不匹配

### 修复方案：强制顺序解码

对于这类视频，**唯一可靠的方案是放弃 seek，使用顺序解码**。

#### 修复 5：扩展检测（同时检查负 PTS 和负 DTS）

```rust
// info.rs
impl StreamInfo {
    pub fn new(...) -> Self {
        // 检查负 PTS 或 DTS（两者都会导致 seek 不可靠）
        let has_negative_pts = frame_times.values()
            .any(|ft| ft.pts < 0 || ft.dts < 0);
        // ...
    }
}
```

#### 修复 6：强制顺序模式

```rust
// lib.rs - get_batch()
let force_sequential = vr.stream_info().has_negative_pts();

let use_sequential = match with_fallback {
    Some(true) => true,
    Some(false) => {
        if force_sequential {
            debug!("Video has negative PTS/DTS - seek is broken, forcing sequential");
            true  // 强制使用顺序模式！
        } else {
            false
        }
    }
    None => vr.should_use_sequential(&indices),
};

// reader.rs - should_use_sequential()
pub fn should_use_sequential(&self, indices: &[usize]) -> bool {
    // 负 PTS/DTS 视频必须使用顺序模式
    if self.stream_info.has_negative_pts() {
        debug!("Video has negative PTS/DTS, must use sequential");
        return true;
    }
    // ... 其他逻辑
}
```

---

## 新增修复：Batch 处理中的状态问题

### 问题 7：EOF 后状态未重置

**症状**：处理视频末尾帧时，先处理的帧触发 EOF，后续帧解码失败。

```
处理 idx=472: 发送 EOF → 成功获取帧
处理 idx=476: 尝试 skip forward → 失败（解码器已 EOF）
```

**修复**：添加 `eof_sent` 标志

```rust
// reader.rs
pub struct VideoReader {
    // ...
    eof_sent: bool,  // 新增：追踪 EOF 状态
}

// can_skip_forward 检查中排除 EOF 状态
let can_skip_forward = !self.eof_sent  // 新增条件！
    && presentation_idx >= self.curr_pres_idx
    && self.curr_pres_idx >= key_pres_idx;

// avflushbuf 中重置 EOF 状态
pub fn avflushbuf(&mut self) -> Result<(), ffmpeg::Error> {
    unsafe { avcodec_flush_buffers(self.decoder.video.as_mut_ptr()) };
    self.eof_sent = false;  // 重置！
    // ...
}
```

### 问题 8：找到目标帧后继续排空缓冲区

**症状**：`get_frame_raw_by_count` 找到目标帧后继续排空解码器，导致 `curr_pres_idx` 跳过后续帧。

```rust
// 修复前：找到帧后继续 while 循环
while self.decoder.video.receive_frame(&mut decoded).is_ok() {
    if self.curr_pres_idx == target_pres_idx {
        result_frame = Some(yuv_frame);
    }
    self.curr_pres_idx += 1;  // 继续增加！
}

// 修复后：找到帧后立即返回
while self.decoder.video.receive_frame(&mut decoded).is_ok() {
    if self.curr_pres_idx == target_pres_idx {
        // ...
        self.curr_pres_idx += 1;
        return (Some(yuv_frame), counter + 1);  // 立即返回！
    }
    self.curr_pres_idx += 1;
}
```

### 问题 9：EOF 处理器使用错误目标

**症状**：`get_frame_raw_after_eof_by_count` 使用 `curr_pres_idx` 而不是原始目标。

```rust
// 修复前
pub fn get_frame_raw_after_eof_by_count(&mut self) -> ... {
    let (yuv_frame, _) = self.get_frame_raw_by_count(self.curr_pres_idx);  // ❌
}

// 修复后：传递正确的目标
pub fn get_frame_raw_after_eof_by_count(
    &mut self, 
    target_pres_idx: usize  // 新增参数！
) -> ... {
    let (yuv_frame, _) = self.get_frame_raw_by_count(target_pres_idx);  // ✅
}
```

---

## 更新后的总结

| 问题 | 原因 | 修复 |
|------|------|------|
| 固定偏移 (+7/-7) | 负 PTS 帧导致 seek 后索引错位 | 计算 `negative_pts_offset` 并条件性应用 |
| 第一个 GOP 正确，后续错误 | 首个 GOP 从头解码无偏移 | 只在 `keyframe > 0` 时应用偏移 |
| 跨 GOP 时错误 | `can_skip_forward` 错误判断 | 增加条件检查 |
| seek 后残留帧 | 滤镜图未刷新 | `avflushbuf` 清空所有缓冲区 |
| **负 PTS 视频 seek 完全失败** | FFmpeg 内部 PTS 归一化 | **强制顺序模式** |
| **负 DTS 视频 seek 到错误位置** | FFmpeg seek 定位错误 | **扩展检测，强制顺序模式** |
| **EOF 后无法继续解码** | 解码器状态未重置 | 添加 `eof_sent` 标志 |
| **找到帧后跳过后续帧** | 排空缓冲区过度 | 找到目标后立即返回 |
| **EOF 处理器目标错误** | 使用 curr_pres_idx | 传递正确的 target_pres_idx |

## 测试视频分类

| 视频 | PTS | DTS | seek 可用 | 推荐模式 | 说明 |
|------|-----|-----|----------|---------|------|
| `0a7ef2bd-*.mp4` | ✅ 正 | ✅ 正 | ✅ | seek (flag=1) | 正常视频 |
| `ea09afcd-*.mov` | ❌ 负 | ❌ 负 | ❌ | **sequential** | FFmpeg PTS 归一化导致映射失效 |
| `out_A_negpts.mp4` | ✅ 正 | ❌ 负 | ✅ | seek (flag=0) | 负 DTS 使用 flag=0 可正常 seek |
| `out_C_negpts.mp4` | ✅ 正 | ❌ 负 | ✅ | seek (flag=0) | 负 DTS 使用 flag=0 可正常 seek |
| `1-9_*.mp4` | ✅ 正 | ❌ 负 | ✅ | seek (flag=0) | 负 DTS 使用 flag=0 可正常 seek |

---

## 2024-12-04 重大发现：负 PTS vs 负 DTS 的本质区别

### 关键发现

1. **FFmpeg 解码器会规范化 PTS**
   - 包 (packet) PTS: -150, -110, -80, ..., 0, -10, 20, ...
   - 解码帧 (frame) PTS: 0, 10, 20, 30, 40, ...
   - FFmpeg 自动加偏移使 min_pts=0

2. **这导致我们的 presentation 映射完全失效**
   - `StreamInfo.presentation_to_decode_idx` 使用包 PTS 构建
   - 但迭代器输出的是规范化后的 PTS 顺序
   - 结果：seek 返回的帧与迭代器帧偏移 N（N=负 PTS 帧数）

3. **负 DTS 不影响解码帧 PTS**
   - 包 DTS 可以为负，但解码帧 PTS 仍从 0 开始
   - 只需使用正确的 seek flag 即可

### 最终处理策略

```
负 PTS 视频:  → 强制顺序模式 (FFmpeg PTS 归一化导致映射失效)
负 DTS 视频:  → Seek + flag=0 + 运行时验证
正常视频:     → Seek + flag=1 (BACKWARD)
```

### 代码实现

```rust
// info.rs
let has_negative_pts = frame_times.values().any(|ft| ft.pts < 0);  // 必须顺序
let has_negative_dts = frame_times.values().any(|ft| ft.dts < 0);  // 可尝试 seek

// reader.rs - needs_sequential_mode()
if self.stream_info.has_negative_pts() {
    return true;  // 必须顺序
}
if self.stream_info.has_negative_dts() {
    return !self.verify_seek_works();  // 验证后决定
}
false  // 正常视频

// reader.rs - seek_frame_by_decode_idx()
let seek_flag = if self.stream_info.has_negative_dts() { 0 } else { 1 };
```

## 最终结论

1. **负 PTS 视频**：**必须使用顺序模式** - FFmpeg PTS 归一化导致 presentation 映射失效
2. **负 DTS 视频**（PTS 非负）：使用 `av_seek_frame` flag=0 可正常 seek
3. **正常视频**（PTS ≥ 0 且 DTS ≥ 0）：使用 flag=1 (BACKWARD) 正常 seek
4. `with_fallback=False` 会自动检测并在必要时切换到顺序模式
5. `with_fallback=None` 综合考虑视频特征和索引分布，自动选择最优方法

### 测试结果 (14/14 通过)

所有测试视频现在都能正确处理，包括：
- 负 PTS + 负 DTS 视频 (`ea09afcd.mov`)
- 仅负 DTS 视频 (`out_A/B/C_negpts.mp4`, `1-9_*.mp4`)
- 正常视频 (`0a7ef2bd.mp4`)

