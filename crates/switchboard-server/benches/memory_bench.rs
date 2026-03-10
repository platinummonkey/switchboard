//! Memory benchmark: idle baseline and 1 000 concurrent SSE streams.
//!
//! Measures resident set size (RSS) before and during a load of
//! 1 000 concurrent streaming connections, verifying the server stays within
//! the target of 100 MB at full load.
//!
//! Run with:
//!   cargo bench -p switchboard-server --bench memory_bench
//!
//! The benchmark does NOT use criterion — it runs as a standalone async binary
//! and prints human-readable results.  It is still compiled as a [[bench]]
//! target so `cargo bench` discovers it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::response::sse::{Event, Sse};
use axum::routing::get;
use futures::stream;
use switchboard_server::observability::UsageTracker;
use switchboard_server::providers::ProviderRegistry;
use switchboard_server::proxy::handler::{AppState, health};

// ── RSS helper ────────────────────────────────────────────────────────────────

/// Return the peak RSS (max resident set size) for this process in bytes.
///
/// Uses `getrusage(RUSAGE_SELF)` — monotonically increasing, so calling it
/// just before and just after the load test gives the correct peak delta.
/// On Linux `ru_maxrss` is in KiB; on macOS it is in bytes.
fn rss_bytes() -> u64 {
    #[cfg(unix)]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        #[cfg(target_os = "macos")]
        {
            ru.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            ru.ru_maxrss as u64 * 1024
        }
    }

    #[cfg(not(unix))]
    {
        0
    }
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1_000_000 {
        format!("{:.1} MB", b as f64 / 1_000_000.0)
    } else {
        format!("{:.1} KB", b as f64 / 1_000.0)
    }
}

// ── Stub server ───────────────────────────────────────────────────────────────

/// Build a minimal AppState for the benchmark (no real providers).
fn bench_app_state() -> Arc<AppState> {
    use switchboard_server::config::ServerConfig;

    Arc::new(AppState {
        config: Arc::new(ServerConfig::default()),
        providers: Arc::new(ProviderRegistry::new()),
        key_pools: Arc::new(HashMap::new()),
        usage: Arc::new(UsageTracker::new()),
        rate_limit_handle: None,
        health_checker: Arc::new(switchboard_server::routing::ProviderHealthChecker::new()),
    })
}

/// An SSE endpoint that streams N events and then closes.
async fn sse_stream_100()
-> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let events = (0..100u32).map(|i| {
        Ok(Event::default().data(format!(
            r#"{{"id":"chunk-{i}","choices":[{{"delta":{{"content":"tok"}}}}]}}"#
        )))
    });
    Sse::new(stream::iter(events))
}

fn bench_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stream", get(sse_stream_100))
        .with_state(state)
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("=== Switchboard memory benchmark ===\n");

    // ── Idle baseline ─────────────────────────────────────────────────────────
    let rss_before = rss_bytes();
    println!("RSS at start        : {}", fmt_bytes(rss_before));

    let state = bench_app_state();
    let router = bench_router(Arc::clone(&state));

    // Bind to a random port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(axum::serve(listener, router).into_future());

    // Allow the server to settle.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let rss_idle = rss_bytes();
    println!("RSS after server start : {}", fmt_bytes(rss_idle));
    println!(
        "Server-startup delta   : +{}",
        fmt_bytes(rss_idle.saturating_sub(rss_before))
    );
    println!();

    // ── 1 000 concurrent SSE streams ─────────────────────────────────────────
    const CONCURRENT: usize = 1_000;
    const CHUNKS_PER_STREAM: usize = 100;

    println!("Opening {CONCURRENT} concurrent SSE streams ({CHUNKS_PER_STREAM} events each)...");

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(CONCURRENT + 10)
        .tcp_keepalive(Duration::from_secs(5))
        .build()
        .unwrap();

    let url = format!("http://{addr}/stream");
    let rss_before_load = rss_bytes();

    // Spawn all streams concurrently; each reads all events then completes.
    let tasks: Vec<_> = (0..CONCURRENT)
        .map(|_| {
            let client = client.clone();
            let url = url.clone();
            tokio::spawn(async move {
                let mut resp = client.get(&url).send().await?;
                let mut count = 0usize;
                while let Some(chunk) = resp.chunk().await? {
                    if !chunk.is_empty() {
                        count += 1;
                    }
                }
                Ok::<usize, reqwest::Error>(count)
            })
        })
        .collect();

    // Measure peak RSS while streams are in-flight.
    let mut peak_rss = rss_before_load;
    let sample_task = tokio::spawn(async move {
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let r = rss_bytes();
            if r > peak_rss {
                peak_rss = r;
            }
        }
        peak_rss
    });

    // Wait for all streams to complete.
    let mut errors = 0usize;
    for task in tasks {
        if task.await.unwrap().is_err() {
            errors += 1;
        }
    }

    let peak_rss = sample_task.await.unwrap();
    let rss_after = rss_bytes();

    println!(
        "Streams completed   : {} ({errors} errors)",
        CONCURRENT - errors
    );
    println!("RSS before load     : {}", fmt_bytes(rss_before_load));
    println!("RSS peak (sampled)  : {}", fmt_bytes(peak_rss));
    println!("RSS after load      : {}", fmt_bytes(rss_after));
    println!(
        "Load delta (peak)   : +{}",
        fmt_bytes(peak_rss.saturating_sub(rss_before_load))
    );
    println!();

    // ── Target check ──────────────────────────────────────────────────────────
    let target_bytes: u64 = 100 * 1_000_000; // 100 MB
    let delta = peak_rss.saturating_sub(rss_before_load);
    if delta <= target_bytes {
        println!("✓ PASS: peak delta {} ≤ target 100 MB", fmt_bytes(delta));
    } else {
        println!(
            "✗ FAIL: peak delta {} > target 100 MB — investigate allocation",
            fmt_bytes(delta)
        );
        std::process::exit(1);
    }
}
