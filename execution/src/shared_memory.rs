//! Shared memory interface for reading signals from Mojo/JAX engine

use std::fs::OpenOptions;
use std::io;

use memmap2::MmapOptions;

use crate::{TradingSignal, NanotronError, Result};

/// Ring buffer header layout
#[repr(C)]
struct RingBufferHeader {
    magic: u64,
    write_pos: u64,
    read_pos: u64,
    record_size: u64,
    max_records: u64,
}

impl RingBufferHeader {
    const MAGIC: u64 = 0x4E414E4F54524F4E; // "NANOTRON"
    const SIZE: usize = std::mem::size_of::<Self>();
}

/// Signal reader from shared memory
pub struct SignalReader {
    mmap: memmap2::Mmap,
    last_read_pos: u64,
}

impl SignalReader {
    /// Create new signal reader
    pub fn new(shm_path: &str) -> Result<Self> {
        // Try to open shared memory file
        let path = format!("/dev/shm{}", shm_path);
        
        let file = match OpenOptions::new().read(true).open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Create dummy for testing
                return Ok(Self::dummy());
            }
            Err(e) => return Err(NanotronError::Io(e)),
        };
        
        // Memory map the file
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(NanotronError::Io)?
        };
        
        Ok(Self {
            mmap,
            last_read_pos: 0,
        })
    }
    
    /// Create dummy reader for testing
    fn dummy() -> Self {
        // Create a minimal valid mmap
        let file = tempfile::tempfile().unwrap();
        file.set_len(1024).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        
        Self {
            mmap,
            last_read_pos: 0,
        }
    }
    
    /// Read next signal (if available)
    pub fn read(&self) -> Option<TradingSignal> {
        if self.mmap.len() < RingBufferHeader::SIZE {
            return None;
        }
        
        // Read header
        let header_bytes = &self.mmap[..RingBufferHeader::SIZE];
        let magic = u64::from_le_bytes(header_bytes[0..8].try_into().ok()?);
        
        if magic != RingBufferHeader::MAGIC {
            return None; // Invalid or not initialized
        }
        
        let write_pos = u64::from_le_bytes(header_bytes[8..16].try_into().ok()?);
        let read_pos = u64::from_le_bytes(header_bytes[16..24].try_into().ok()?);
        let record_size = u64::from_le_bytes(header_bytes[24..32].try_into().ok()?);
        let max_records = u64::from_le_bytes(header_bytes[32..40].try_into().ok()?);
        
        if read_pos >= write_pos {
            return None; // No new data
        }
        
        // Calculate offset
        let offset = RingBufferHeader::SIZE + 
            ((read_pos % max_records) as usize * record_size as usize);
        
        let end_offset = offset + TradingSignal::SIZE;
        if end_offset > self.mmap.len() {
            return None;
        }
        
        // Read signal
        let signal_bytes = &self.mmap[offset..end_offset];
        TradingSignal::from_bytes(signal_bytes)
    }
    
    /// Check if new data is available
    pub fn has_data(&self) -> bool {
        if self.mmap.len() < RingBufferHeader::SIZE {
            return false;
        }
        
        let header_bytes = &self.mmap[..RingBufferHeader::SIZE];
        let write_pos = u64::from_le_bytes(
            header_bytes[8..16].try_into().unwrap_or([0; 8])
        );
        let read_pos = u64::from_le_bytes(
            header_bytes[16..24].try_into().unwrap_or([0; 8])
        );
        
        write_pos > read_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_serialization() {
        let signal = TradingSignal {
            ticker_id: 42,
            direction: 1,
            confidence: 0.85,
            size: 1000.0,
            reasoning_depth: 8,
            latency_us: 50,
            timestamp_ns: 0,
        };
        
        let bytes = signal.to_bytes();
        let signal2 = TradingSignal::from_bytes(&bytes).unwrap();
        
        assert_eq!(signal.ticker_id, signal2.ticker_id);
        assert_eq!(signal.direction, signal2.direction);
        assert!((signal.confidence - signal2.confidence).abs() < 1e-6);
    }
}

