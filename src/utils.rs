use ffmpeg_next::format::Pixel as AvPixel;
use ffmpeg_next::{format::context::Input, Error, Packet};
use ndarray::{Array3, Array4, ArrayViewMut3};
use std::sync::OnceLock;

pub type FrameArray = Array3<u8>;
pub type VideoArray = Array4<u8>;

/// Always use NV12 pixel format with hardware acceleration, then rescale later.
pub(crate) static HWACCEL_PIXEL_FORMAT: AvPixel = AvPixel::NV12;

pub fn init_ffmpeg() -> Result<(), Error> {
    // ffmpeg-next initializes a mutable error-string table; constructors can run concurrently.
    static INIT: OnceLock<Result<(), Error>> = OnceLock::new();
    *INIT.get_or_init(ffmpeg_next::init)
}

/// Recoverable read failures allowed without top-level progress. Nested demuxers
/// (hls, concat) advance their own AVIO while the playlist's position stays put.
const MAX_STALLED_READS: u32 = 1000;

/// Preserve recoverable demuxer errors without mistaking an I/O failure for EOF.
pub fn read_packet(input: &mut Input) -> Result<Option<Packet>, Error> {
    let mut packet = Packet::empty();
    let (mut last_position, mut stalls) = (-1, 0);
    loop {
        match packet.read(input) {
            Ok(()) => return Ok(Some(packet)),
            Err(Error::Eof) => return Ok(None),
            Err(
                error @ (Error::InvalidData
                | Error::PatchWelcome
                | Error::Other {
                    errno: ffmpeg_next::error::EAGAIN,
                }),
            ) => {
                // SAFETY: input owns this context; no read callback is running here.
                unsafe {
                    let io = (*input.as_mut_ptr()).pb;
                    if io.is_null() {
                        return Err(error);
                    }
                    if (*io).error != 0 {
                        return Err(Error::from((*io).error));
                    }
                    // avio_tell is an inline C wrapper around this seek.
                    let position = ffmpeg_next::ffi::avio_seek(io, 0, libc::SEEK_CUR);
                    if position > last_position {
                        (last_position, stalls) = (position, 0);
                    } else if stalls == MAX_STALLED_READS {
                        return Err(error);
                    } else {
                        stalls += 1;
                    }
                }
            }
            Err(error) => return Err(error),
        }
    }
}

pub fn insert_frame(frame_array: &mut ArrayViewMut3<u8>, frame: FrameArray) {
    frame_array.zip_mut_with(&frame, |a, b| {
        *a = *b;
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{self, Cursor, Read, Seek, SeekFrom};
    use std::sync::{
        atomic::{AtomicI32, Ordering},
        mpsc, Arc,
    };
    use std::time::Duration;

    pub(crate) struct FailingRead {
        pub data: Cursor<Vec<u8>>,
        pub error: Arc<AtomicI32>,
    }

    impl Read for FailingRead {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let error = self.error.load(Ordering::SeqCst);
            if error != 0 {
                return Err(io::Error::from(if error == ffmpeg_next::error::EAGAIN {
                    io::ErrorKind::WouldBlock
                } else {
                    io::ErrorKind::Other
                }));
            }
            let size = buffer.len().min(1024);
            self.data.read(&mut buffer[..size])
        }
    }

    impl Seek for FailingRead {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.data.seek(position)
        }
    }

    #[test]
    fn packet_reads_surface_persistent_io_errors() {
        init_ffmpeg().unwrap();
        for errno in [ffmpeg_next::error::EIO, ffmpeg_next::error::EAGAIN] {
            let error = Arc::new(AtomicI32::new(0));
            let stream = FailingRead {
                data: Cursor::new(std::fs::read("assets/input.mp4").unwrap()),
                error: error.clone(),
            };
            let io = ffmpeg_next::format::context::StreamIo::from_read_seek(stream).unwrap();
            let mut input = ffmpeg_next::format::input_from_stream(io, None, None).unwrap();
            input.seek(0, ..10).unwrap();
            error.store(errno, Ordering::SeqCst);
            let (sender, receiver) = mpsc::channel();
            std::thread::spawn(move || loop {
                match read_packet(&mut input) {
                    Ok(Some(_)) => (),
                    result => {
                        sender.send(result.map(|_| ())).unwrap();
                        break;
                    }
                }
            });
            let result = receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("packet reads must not loop on a persistent I/O error");
            assert_eq!(result, Err(Error::Other { errno }));
        }
    }
}
