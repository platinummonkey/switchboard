//! Postgres persistence layer for switchboard-server.
//!
//! [`DbPool`] wraps a write (primary) pool and an optional set of read-replica
//! pools, round-robining across replicas for read queries.  When no read
//! replicas are configured, reads fall back to the write pool.

pub mod queries;
pub mod schema;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use diesel_async::AsyncPgConnection;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;
use diesel_async::pooled_connection::deadpool::Pool;
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};

use crate::config::DatabaseConfig;
use crate::error::ServerError;

const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

/// A type alias for a deadpool pooled connection.
pub type PooledConn = diesel_async::pooled_connection::deadpool::Object<AsyncPgConnection>;

/// Write + read-replica connection pool.
pub struct DbPool {
    write: Pool<AsyncPgConnection>,
    read: Vec<Pool<AsyncPgConnection>>,
    read_counter: Arc<AtomicUsize>,
}

impl DbPool {
    /// Connect to Postgres using the supplied [`DatabaseConfig`].
    ///
    /// Fails fast if any pool cannot be established — this is intentional:
    /// when `database.enabled = true` the server should not start with a broken
    /// DB connection.
    pub async fn connect(cfg: &DatabaseConfig) -> Result<Self, ServerError> {
        let write = {
            let manager = AsyncDieselConnectionManager::<AsyncPgConnection>::new(&cfg.write_url);
            Pool::builder(manager)
                .max_size(cfg.max_connections as usize)
                .build()
                .map_err(|e| ServerError::Database(format!("write pool connect: {e}")))?
        };

        let mut read = Vec::with_capacity(cfg.read_urls.len());
        for url in &cfg.read_urls {
            let manager = AsyncDieselConnectionManager::<AsyncPgConnection>::new(url);
            let pool = Pool::builder(manager)
                .max_size(cfg.max_connections as usize)
                .build()
                .map_err(|e| ServerError::Database(format!("read pool connect ({url}): {e}")))?;
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

    /// Return a pooled connection from the write (primary) pool.
    pub async fn write(&self) -> Result<PooledConn, ServerError> {
        self.write
            .get()
            .await
            .map_err(|e| ServerError::Database(format!("write pool acquire: {e}")))
    }

    /// Return a pooled connection from a read pool, round-robining across
    /// replicas. Falls back to the write pool when no replicas are configured.
    pub async fn read(&self) -> Result<PooledConn, ServerError> {
        if self.read.is_empty() {
            return self.write().await;
        }
        let idx = self.read_counter.fetch_add(1, Ordering::Relaxed) % self.read.len();
        self.read[idx]
            .get()
            .await
            .map_err(|e| ServerError::Database(format!("read pool acquire: {e}")))
    }

    /// Run all pending migrations against the write pool.
    ///
    /// Uses a blocking `diesel::PgConnection` via `spawn_blocking` because
    /// `MigrationHarness` does not support async connections.
    pub async fn migrate(&self, write_url: &str) -> Result<(), ServerError> {
        let url = write_url.to_string();
        tokio::task::spawn_blocking(move || {
            use diesel::Connection;
            let mut conn = diesel::PgConnection::establish(&url)
                .map_err(|e| ServerError::Database(format!("migration connect: {e}")))?;
            conn.run_pending_migrations(MIGRATIONS)
                .map_err(|e| ServerError::Database(format!("migration failed: {e}")))?;
            Ok::<_, ServerError>(())
        })
        .await
        .map_err(|e| ServerError::Database(format!("spawn_blocking join error: {e}")))??;
        tracing::info!("database migrations applied");
        Ok(())
    }
}
