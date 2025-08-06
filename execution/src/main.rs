//! Project Nanotron — Rust Backend Server
//!
//! Streams typed messages from the strategy core (via shared memory) to
//! browser clients over WebSocket. Exposes JSON status and Prometheus
//! metrics for the operator dashboard and for k8s liveness/readiness.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{ws::WebSocketUpgrade, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use nanotron_backend::{
    shared_memory::SignalReader,
    websocket::{handle_socket, WsMessage},
    AppState, Metrics,
};

const DEFAULT_BIND: &str = "0.0.0.0:8080";
const SHM_PATH: &str = "/nanotron_signals";
const BROADCAST_CAPACITY: usize = 1024;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let bind: SocketAddr = std::env::var("NANOTRON_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND.into())
        .parse()?;

    info!("starting nanotron backend on {bind}");

    let metrics = Arc::new(Metrics::new());
    let signal_reader = Arc::new(SignalReader::new(SHM_PATH)?);
    if !signal_reader.is_attached() {
        info!("producer at /dev/shm{SHM_PATH} not present — server starts in detached mode");
    }

    let (tx, _rx) = broadcast::channel::<WsMessage>(BROADCAST_CAPACITY);

    spawn_signal_pump(tx.clone(), signal_reader.clone(), metrics.clone());
    spawn_status_pump(tx.clone(), metrics.clone());

    let state = Arc::new(AppState {
        metrics: metrics.clone(),
        signal_reader: signal_reader.clone(),
        broadcast_tx: tx,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/status", get(status))
        .route("/metrics", get(metrics_endpoint))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!("listening on {bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn spawn_signal_pump(
    tx: broadcast::Sender<WsMessage>,
    reader: Arc<SignalReader>,
    metrics: Arc<Metrics>,
) {
    tokio::spawn(async move {
        // Adaptive backoff: tight loop while data is flowing, exponential
        // back-off up to 5 ms when the producer is idle. Avoids burning
        // CPU when detached or quiet without stretching the latency tail
        // when active.
        let mut idle_us: u64 = 100;
        loop {
            if let Some(signal) = reader.read() {
                metrics.record_signal(signal.latency_us);
                idle_us = 100;
                if signal.direction != 0 {
                    let _ = tx.send(WsMessage::Signal(signal));
                }
            } else {
                tokio::time::sleep(tokio::time::Duration::from_micros(idle_us)).await;
                idle_us = (idle_us * 2).min(5_000);
            }
        }
    });
}

fn spawn_status_pump(tx: broadcast::Sender<WsMessage>, metrics: Arc<Metrics>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            tick.tick().await;
            let _ = tx.send(WsMessage::Status(metrics.get_status()));
        }
    });
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => info!("ctrl-c received"),
        _ = term => info!("SIGTERM received"),
    }
}

async fn index() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "Nanotron Backend",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": ["/health", "/ready", "/status", "/metrics", "/ws"]
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "healthy" }))
}

async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let attached = state.signal_reader.is_attached();
    let body = serde_json::json!({
        "ready": attached,
        "producer_attached": attached,
    });
    if attached {
        (axum::http::StatusCode::OK, Json(body))
    } else {
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(body))
    }
}

async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.get_status())
}

async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        state.metrics.render_prometheus(),
    )
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.broadcast_tx.subscribe();
    let metrics = state.metrics.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, rx, metrics))
}
