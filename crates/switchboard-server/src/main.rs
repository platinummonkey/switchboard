#[allow(dead_code, unused_imports)]
mod auth;
#[allow(dead_code, unused_imports)]
mod config;
#[allow(dead_code, unused_imports)]
mod error;
#[allow(dead_code, unused_imports)]
mod guardrails;
#[allow(dead_code, unused_imports)]
mod identity;
#[allow(dead_code, unused_imports)]
mod key_pool;
#[allow(dead_code, unused_imports)]
mod middleware;
#[allow(dead_code, unused_imports)]
mod routing;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("switchboard-server starting");
    Ok(())
}
