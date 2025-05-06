//! Project Nanotron — Rust Backend Server
//!
//! WebSocket server for streaming data to frontend.
//! Reads from shared memory (signals from Mojo/JAX engine).

use std::sync::Arc;
use std::net::SocketAddr;

use axum::{
    routing::get,
    Router,
    extract::{State, ws::{WebSocket, WebSocketUpgrade, Message}},
    response::IntoResponse,
    Json,
};
use tower_http::cors::{CorsLayer, Any};
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;
use tokio::sync::broadcast;
use futures_util::{StreamExt, SinkExt};

use nanotron_backend::{
    AppState, Metrics, TradingSignal, EngineStatus, OrderBook, PortfolioSummary,
    shared_memory::SignalReader,
};

/// WebSocket message types
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
enum WsMessage {
    Signal(TradingSignal),
    OrderBook(OrderBook),
    Status(EngineStatus),
    Portfolio(PortfolioSummary),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Nanotron Backend Server...");
    
    // Initialize metrics
    let metrics = Arc::new(Metrics::new());
    
    // Initialize signal reader
    let signal_reader = Arc::new(SignalReader::new("/nanotron_signals")?);
    
    // Create broadcast channel for WebSocket clients
    let (tx, _rx) = broadcast::channel::<WsMessage>(1000);
    let tx = Arc::new(tx);
    
    // Spawn signal reading task
    let tx_clone = tx.clone();
    let signal_reader_clone = signal_reader.clone();
    let metrics_clone = metrics.clone();
    
    tokio::spawn(async move {
        loop {
            if let Some(signal) = signal_reader_clone.read() {
                metrics_clone.record_signal(signal.latency_us);
                
                if signal.direction != 0 {
                    let _ = tx_clone.send(WsMessage::Signal(signal));
                }
            }
            
            // Small yield to avoid busy loop
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
    });
    
    // Spawn status update task
    let tx_clone = tx.clone();
    let metrics_clone = metrics.clone();
    
    tokio::spawn(async move {
        loop {
            let status = metrics_clone.get_status();
            let _ = tx_clone.send(WsMessage::Status(status));
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
    
    // Create app state
    let state = Arc::new(AppState {
        metrics: metrics.clone(),
        signal_reader,
    });
    
    // Build router
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/ws", get(|ws: WebSocketUpgrade| async move {
            ws.on_upgrade(|socket| handle_websocket(socket, tx.subscribe()))
        }))
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any));
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn index() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "Nanotron Backend",
        "version": "0.1.0",
        "endpoints": ["/health", "/status", "/ws"]
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "healthy" }))
}

async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.get_status())
}

async fn handle_websocket(
    socket: WebSocket,
    mut rx: broadcast::Receiver<WsMessage>,
) {
    let (mut sender, mut receiver) = socket.split();
    
    // Spawn task to forward messages to client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });
    
    // Handle incoming messages from client
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                info!("Received from client: {}", text);
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    
    send_task.abort();
}

