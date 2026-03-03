//! Postgres persistence layer for switchboard-server.
//!
//! [`DbPool`] wraps a write (primary) pool and an optional set of read-replica
//! pools, round-robining across replicas for read queries.  When no read
//! replicas are configured, reads fall back to the write pool.

pub mod queries;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use crate::config::DatabaseConfig;
use crate::error::ServerError;

/// Write + read-replica connection pool.
pub struct DbPool {
    write: PgPool,
    read: Vec<PgPool>,
    read_counter: Arc<AtomicUsize>,
}

impl DbPool {
    /// Connect to Postgres using the supplied [`DatabaseConfig`].
    ///
    /// Fails fast if any pool cannot be established — this is intentional:
    /// when `database.enabled = true` the server should not start with a broken
    /// DB connection.
    pub async fn connect(cfg: &DatabaseConfig) -> Result<Self, ServerError> {
        let timeout = crate::config::duration::parse(&cfg.connection_timeout)
            .unwrap_or(std::time::Duration::from_secs(5));

        let write = PgPoolOptions::new()
            .max_connections(cfg.max_connections)
            .acquire_timeout(timeout)
            .connect(&cfg.write_url)
            .await
            .map_err(|e| ServerError::Database(format!("write pool connect: {e}")))?;

        let mut read = Vec::with_capacity(cfg.read_urls.len());
        for url in &cfg.read_urls {
            let pool = PgPoolOptions::new()
                .max_connections(cfg.max_connections)
                .acquire_timeout(timeout)
                .connect(url)
                .await
                .map_err(|e| {
                    ServerError::Database(format!("read pool connect ({url}): {e}"))
                })?;
            read.push(pool);
        }

        tracing::info!(
            read_replicas = read.len(),
            max_connections = cfg.max_connections,
            "database connection pools established"
        );

        Ok(Self {
            write,
            read,
            read_counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Return a reference to the write (primary) pool.
    pub fn write(&self) -> &PgPool {
        &self.write
    }

    /// Return a reference to a read pool, round-robining across replicas.
    /// Falls back to the write pool when no replicas are configured.
    pub fn read(&self) -> &PgPool {
        if self.read.is_empty() {
            return &self.write;
        }
        let idx = self.read_counter.fetch_add(1, Ordering::Relaxed) % self.read.len();
        &self.read[idx]
    }

    /// Run all pending migrations against the write pool.
    pub async fn migrate(&self) -> Result<(), ServerError> {
        sqlx::migrate!("./migrations")
            .run(&self.write)
            .await
            .map_err(|e| ServerError::Database(format!("migration failed: {e}")))?;
        tracing::info!("database migrations applied");
        Ok(())
    }
}
