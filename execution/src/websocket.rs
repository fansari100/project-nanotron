//! WebSocket fan-out from the broadcast channel to connected clients.
//!
//! The signal-reader task in `main.rs` produces typed `WsMessage`s into a
//! `tokio::sync::broadcast` channel. Each subscriber gets its own
//! `Receiver`; a slow client that lags more than the channel capacity
//! gets `RecvError::Lagged` and we drop the connection instead of letting
//! it back-pressure the producer.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::{EngineStatus, OrderBook, PortfolioSummary, TradingSignal};
use crate::metrics::Metrics;

/// Discriminated union of everything that can flow over the wire.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    Signal(TradingSignal),
    OrderBook(OrderBook),
    Status(EngineStatus),
    Portfolio(PortfolioSummary),
}

/// Subscribe a freshly-upgraded socket to the broadcast channel and forward
/// each message until either side hangs up.
pub async fn handle_socket(
    socket: WebSocket,
    mut rx: broadcast::Receiver<WsMessage>,
    metrics: Arc<Metrics>,
) {
    metrics.ws_client_connected();
    let (mut sender, mut receiver) = socket.split();

    let send_metrics = metrics.clone();
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    let json = match serde_json::to_string(&msg) {
                        Ok(s) => s,
                        Err(e) => {
                            send_metrics.ws_send_error();
                            warn!("serialize ws msg failed: {e}");
                            continue;
                        }
                    };
                    if sender.send(Message::Text(json)).await.is_err() {
                        send_metrics.ws_send_error();
                        break;
                    }
                    send_metrics.ws_message_sent();
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    warn!("ws client lagged, skipped {skipped} messages — closing");
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => debug!("client text: {text}"),
            Message::Ping(_) | Message::Pong(_) => {}
            Message::Close(_) => break,
            Message::Binary(_) => {}
        }
    }

    send_task.abort();
    metrics.ws_client_disconnected();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_message_tag_is_external() {
        let msg = WsMessage::Status(EngineStatus {
            running: true,
            signals_generated: 0,
            orders_sent: 0,
            avg_latency_us: 0.0,
            uptime_seconds: 0,
        });
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.starts_with(r#"{"type":"Status""#), "actual: {s}");
    }
}
