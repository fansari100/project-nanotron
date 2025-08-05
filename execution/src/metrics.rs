//! Process metrics — both the engine status used by `/status` and the
//! Prometheus text exposition exposed at `/metrics`.
//!
//! Counters are `AtomicU64` so they're cheap to bump from the signal-reader
//! task without a lock; the histogram uses Atkinson's running-quantile
//! approximation so we can publish a p50/p99 without locking either.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::EngineStatus;

/// Lock-free counter set + a small running-quantile estimator for latency.
pub struct Metrics {
    pub signals_count: AtomicU64,
    pub orders_count: AtomicU64,
    pub ws_clients: AtomicU64,
    pub ws_messages_sent: AtomicU64,
    pub ws_send_errors: AtomicU64,
    latency_us: Mutex<RunningQuantile>,
    start_time: Instant,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            signals_count: AtomicU64::new(0),
            orders_count: AtomicU64::new(0),
            ws_clients: AtomicU64::new(0),
            ws_messages_sent: AtomicU64::new(0),
            ws_send_errors: AtomicU64::new(0),
            latency_us: Mutex::new(RunningQuantile::new()),
            start_time: Instant::now(),
        }
    }

    pub fn record_signal(&self, latency_us: i64) {
        self.signals_count.fetch_add(1, Ordering::Relaxed);
        if latency_us >= 0 {
            if let Ok(mut q) = self.latency_us.lock() {
                q.observe(latency_us as f64);
            }
        }
    }

    pub fn record_order(&self) {
        self.orders_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_client_connected(&self) {
        self.ws_clients.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_client_disconnected(&self) {
        // saturating sub via compare-exchange loop
        let mut current = self.ws_clients.load(Ordering::Relaxed);
        while current > 0 {
            match self.ws_clients.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(c) => current = c,
            }
        }
    }

    pub fn ws_message_sent(&self) {
        self.ws_messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_send_error(&self) {
        self.ws_send_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot for the JSON `/status` endpoint.
    pub fn get_status(&self) -> EngineStatus {
        let signals = self.signals_count.load(Ordering::Relaxed);
        let orders = self.orders_count.load(Ordering::Relaxed);

        let avg_latency_us = self
            .latency_us
            .lock()
            .ok()
            .map(|q| q.mean())
            .unwrap_or(0.0);

        EngineStatus {
            running: true,
            signals_generated: signals,
            orders_sent: orders,
            avg_latency_us,
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    /// Prometheus text exposition (v0.0.4 plain text).
    pub fn render_prometheus(&self) -> String {
        let signals = self.signals_count.load(Ordering::Relaxed);
        let orders = self.orders_count.load(Ordering::Relaxed);
        let clients = self.ws_clients.load(Ordering::Relaxed);
        let msgs = self.ws_messages_sent.load(Ordering::Relaxed);
        let errs = self.ws_send_errors.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed().as_secs();

        let (mean, p50, p99) = self
            .latency_us
            .lock()
            .ok()
            .map(|q| (q.mean(), q.p(0.5), q.p(0.99)))
            .unwrap_or((0.0, 0.0, 0.0));

        let mut s = String::with_capacity(2048);
        s.push_str("# HELP nanotron_signals_total Signals read from shared memory.\n");
        s.push_str("# TYPE nanotron_signals_total counter\n");
        s.push_str(&format!("nanotron_signals_total {}\n", signals));

        s.push_str("# HELP nanotron_orders_total Orders sent.\n");
        s.push_str("# TYPE nanotron_orders_total counter\n");
        s.push_str(&format!("nanotron_orders_total {}\n", orders));

        s.push_str("# HELP nanotron_ws_clients Currently connected websocket clients.\n");
        s.push_str("# TYPE nanotron_ws_clients gauge\n");
        s.push_str(&format!("nanotron_ws_clients {}\n", clients));

        s.push_str("# HELP nanotron_ws_messages_sent_total Websocket messages forwarded.\n");
        s.push_str("# TYPE nanotron_ws_messages_sent_total counter\n");
        s.push_str(&format!("nanotron_ws_messages_sent_total {}\n", msgs));

        s.push_str("# HELP nanotron_ws_send_errors_total Websocket send failures.\n");
        s.push_str("# TYPE nanotron_ws_send_errors_total counter\n");
        s.push_str(&format!("nanotron_ws_send_errors_total {}\n", errs));

        s.push_str("# HELP nanotron_uptime_seconds Process uptime.\n");
        s.push_str("# TYPE nanotron_uptime_seconds gauge\n");
        s.push_str(&format!("nanotron_uptime_seconds {}\n", uptime));

        s.push_str("# HELP nanotron_signal_latency_us Signal age at read time (microseconds).\n");
        s.push_str("# TYPE nanotron_signal_latency_us summary\n");
        s.push_str(&format!("nanotron_signal_latency_us{{quantile=\"0.5\"}} {:.3}\n", p50));
        s.push_str(&format!("nanotron_signal_latency_us{{quantile=\"0.99\"}} {:.3}\n", p99));
        s.push_str(&format!("nanotron_signal_latency_us_sum {:.3}\n", mean * signals as f64));
        s.push_str(&format!("nanotron_signal_latency_us_count {}\n", signals));

        s
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// P-square quantile estimator (Jain & Chlamtac 1985).
///
/// Constant memory per quantile, no full sample buffer. Accuracy degrades
/// for adversarial inputs but is more than enough for ops dashboards on
/// per-second latency streams.
struct RunningQuantile {
    n: u64,
    sum: f64,
    p50: PSquare,
    p99: PSquare,
}

impl RunningQuantile {
    fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            p50: PSquare::new(0.5),
            p99: PSquare::new(0.99),
        }
    }

    fn observe(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.p50.observe(x);
        self.p99.observe(x);
    }

    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    fn p(&self, q: f64) -> f64 {
        if (q - 0.5).abs() < 1e-9 {
            self.p50.estimate()
        } else if (q - 0.99).abs() < 1e-9 {
            self.p99.estimate()
        } else {
            self.mean()
        }
    }
}

struct PSquare {
    p: f64,
    n: [f64; 5],
    np: [f64; 5],
    dn: [f64; 5],
    q: [f64; 5],
    count: u64,
}

impl PSquare {
    fn new(p: f64) -> Self {
        Self {
            p,
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            q: [0.0; 5],
            count: 0,
        }
    }

    fn observe(&mut self, x: f64) {
        self.count += 1;
        if self.count <= 5 {
            self.q[(self.count - 1) as usize] = x;
            if self.count == 5 {
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            return;
        }

        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        for i in 1..4 {
            let d = self.np[i] - self.n[i];
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                let s = d.signum();
                let qp = self.parabolic(i, s);
                self.q[i] = if self.q[i - 1] < qp && qp < self.q[i + 1] {
                    qp
                } else {
                    self.linear(i, s)
                };
                self.n[i] += s;
            }
        }
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qip = self.q[i + 1];
        let qim = self.q[i - 1];
        let ni = self.n[i];
        let nip = self.n[i + 1];
        let nim = self.n[i - 1];
        qi + d / (nip - nim)
            * ((ni - nim + d) * (qip - qi) / (nip - ni)
                + (nip - ni - d) * (qi - qim) / (ni - nim))
    }

    fn linear(&self, i: usize, d: f64) -> f64 {
        let j = (i as isize + d as isize) as usize;
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i])
    }

    fn estimate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        if self.count <= 5 {
            let mut sorted = self.q;
            let len = self.count as usize;
            let slice = &mut sorted[..len];
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((len as f64 - 1.0) * self.p).round() as usize;
            return slice[idx];
        }
        self.q[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment_under_concurrent_load() {
        use std::sync::Arc;
        let m = Arc::new(Metrics::new());
        let mut handles = vec![];
        for _ in 0..8 {
            let m = m.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..1000 {
                    m.record_signal(i as i64);
                    m.record_order();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.signals_count.load(Ordering::Relaxed), 8000);
        assert_eq!(m.orders_count.load(Ordering::Relaxed), 8000);
    }

    #[test]
    fn prometheus_text_well_formed() {
        let m = Metrics::new();
        m.record_signal(50);
        m.record_signal(200);
        m.record_order();
        let s = m.render_prometheus();
        assert!(s.contains("nanotron_signals_total 2"));
        assert!(s.contains("nanotron_orders_total 1"));
        assert!(s.contains("# TYPE nanotron_uptime_seconds gauge"));
    }

    #[test]
    fn quantile_estimator_within_bounds() {
        let mut q = RunningQuantile::new();
        for i in 1..=1000 {
            q.observe(i as f64);
        }
        let p50 = q.p(0.5);
        let p99 = q.p(0.99);
        assert!(p50 > 400.0 && p50 < 600.0, "p50 = {}", p50);
        assert!(p99 > 950.0 && p99 < 1010.0, "p99 = {}", p99);
    }

    #[test]
    fn ws_client_counter_saturates_at_zero() {
        let m = Metrics::new();
        m.ws_client_disconnected();
        m.ws_client_disconnected();
        assert_eq!(m.ws_clients.load(Ordering::Relaxed), 0);
        m.ws_client_connected();
        m.ws_client_disconnected();
        m.ws_client_disconnected();
        assert_eq!(m.ws_clients.load(Ordering::Relaxed), 0);
    }
}
