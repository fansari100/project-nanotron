//! Project Nanotron — Rust Backend Library
//! 
//! Provides:
//! - Shared memory interface for reading signals
//! - WebSocket streaming to frontend
//! - Metrics collection

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

pub mod shared_memory;
pub mod websocket;
pub mod metrics;

/// Error types for Nanotron backend
#[derive(Error, Debug)]
pub enum NanotronError {
    #[error("Shared memory error: {0}")]
    SharedMemory(String),
    
    #[error("WebSocket error: {0}")]
    WebSocket(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, NanotronError>;

/// Trading signal from the engine
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TradingSignal {
    pub ticker_id: u32,
    pub direction: i8,
    pub confidence: f32,
    pub size: f32,
    pub reasoning_depth: i32,
    pub latency_us: i64,
    pub timestamp_ns: u64,
}

impl TradingSignal {
    pub const SIZE: usize = 32;
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        
        Some(Self {
            ticker_id: u32::from_le_bytes(bytes[0..4].try_into().ok()?),
            direction: bytes[4] as i8,
            // padding: bytes[5..8]
            confidence: f32::from_le_bytes(bytes[8..12].try_into().ok()?),
            size: f32::from_le_bytes(bytes[12..16].try_into().ok()?),
            reasoning_depth: i32::from_le_bytes(bytes[16..20].try_into().ok()?),
            latency_us: i64::from_le_bytes(bytes[20..28].try_into().ok()?),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }
    
    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.ticker_id.to_le_bytes());
        bytes[4] = self.direction as u8;
        // padding: bytes[5..8]
        bytes[8..12].copy_from_slice(&self.confidence.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.size.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.reasoning_depth.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.latency_us.to_le_bytes());
        bytes
    }
    
    /// Get direction as string
    pub fn direction_str(&self) -> &'static str {
        match self.direction {
            -1 => "SELL",
            0 => "HOLD",
            1 => "BUY",
            _ => "UNKNOWN",
        }
    }
}

/// Order book level
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: i32,
    pub order_count: i32,
}

/// Full order book snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp_ns: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    pub fn mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        (self.bids[0].price + self.asks[0].price) / 2.0
    }
    
    pub fn spread(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        self.asks[0].price - self.bids[0].price
    }
    
    pub fn order_imbalance(&self) -> f64 {
        let total_bid: i32 = self.bids.iter().map(|l| l.size).sum();
        let total_ask: i32 = self.asks.iter().map(|l| l.size).sum();
        let total = total_bid + total_ask;
        if total == 0 {
            return 0.0;
        }
        (total_bid - total_ask) as f64 / total as f64
    }
}

/// Position tracking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Portfolio summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PortfolioSummary {
    pub total_value: f64,
    pub cash: f64,
    pub positions: Vec<Position>,
    pub daily_pnl: f64,
    pub sharpe_ratio: f64,
}

/// Engine status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineStatus {
    pub running: bool,
    pub signals_generated: u64,
    pub orders_sent: u64,
    pub avg_latency_us: f64,
    pub uptime_seconds: u64,
}

/// Global metrics
pub struct Metrics {
    pub signals_count: AtomicU64,
    pub orders_count: AtomicU64,
    pub latency_sum_us: AtomicU64,
    pub start_time: std::time::Instant,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            signals_count: AtomicU64::new(0),
            orders_count: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn record_signal(&self, latency_us: i64) {
        self.signals_count.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_us.fetch_add(latency_us as u64, Ordering::Relaxed);
    }
    
    pub fn record_order(&self) {
        self.orders_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_status(&self) -> EngineStatus {
        let signals = self.signals_count.load(Ordering::Relaxed);
        let orders = self.orders_count.load(Ordering::Relaxed);
        let latency_sum = self.latency_sum_us.load(Ordering::Relaxed);
        
        EngineStatus {
            running: true,
            signals_generated: signals,
            orders_sent: orders,
            avg_latency_us: if signals > 0 {
                latency_sum as f64 / signals as f64
            } else {
                0.0
            },
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Application state
pub struct AppState {
    pub metrics: Arc<Metrics>,
    pub signal_reader: Arc<shared_memory::SignalReader>,
}

