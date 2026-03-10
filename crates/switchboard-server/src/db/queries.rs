//! Typed query functions for the three persistence tables.
//!
//! All queries use runtime-checked `sqlx::query_as` so that a live Postgres
//! instance is not required during compilation or in CI.

use sqlx::PgPool;

use crate::error::ServerError;

// ── Row types ──────────────────────────────────────────────────────────────────

/// A row from the `key_pool_entries` table.
#[derive(sqlx::FromRow)]
pub struct KeyPoolEntryRow {
    pub provider_id: String,
    pub id: String,
    pub key_type: String,
    pub weight: f64,
    pub status: String,
    pub source_config: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// A row from the `rate_limit_overrides` table.
#[derive(sqlx::FromRow)]
pub struct RateLimitOverrideRow {
    pub id: String,
    pub rpm: i32,
    pub tpm: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// A row from the `config_overrides` table.
#[derive(sqlx::FromRow)]
pub struct ConfigOverrideRow {
    pub section: String,
    pub config_json: serde_json::Value,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

// ── key_pool_entries ──────────────────────────────────────────────────────────

pub async fn list_key_pool_entries(pool: &PgPool) -> Result<Vec<KeyPoolEntryRow>, ServerError> {
    sqlx::query_as::<_, KeyPoolEntryRow>(
        "SELECT provider_id, id, key_type, weight, status, source_config, created_at, updated_at \
         FROM key_pool_entries \
         ORDER BY provider_id, id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_key_pool_entry(
    pool: &PgPool,
    provider_id: &str,
    id: &str,
    key_type: &str,
    weight: f64,
    status: &str,
    source_config: &serde_json::Value,
) -> Result<(), ServerError> {
    sqlx::query(
        "INSERT INTO key_pool_entries (provider_id, id, key_type, weight, status, source_config, updated_at) \
         VALUES ($1, $2, $3, $4, $5, $6, NOW()) \
         ON CONFLICT (provider_id, id) DO UPDATE SET \
             key_type = EXCLUDED.key_type, \
             weight = EXCLUDED.weight, \
             status = EXCLUDED.status, \
             source_config = EXCLUDED.source_config, \
             updated_at = NOW()",
    )
    .bind(provider_id)
    .bind(id)
    .bind(key_type)
    .bind(weight)
    .bind(status)
    .bind(source_config)
    .execute(pool)
    .await
    .map(|_| ())
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn update_key_weight(
    pool: &PgPool,
    provider_id: &str,
    id: &str,
    weight: f64,
) -> Result<(), ServerError> {
    sqlx::query(
        "UPDATE key_pool_entries SET weight = $1, updated_at = NOW() \
         WHERE provider_id = $2 AND id = $3",
    )
    .bind(weight)
    .bind(provider_id)
    .bind(id)
    .execute(pool)
    .await
    .map(|_| ())
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn update_key_status(
    pool: &PgPool,
    provider_id: &str,
    id: &str,
    status: &str,
) -> Result<(), ServerError> {
    sqlx::query(
        "UPDATE key_pool_entries SET status = $1, updated_at = NOW() \
         WHERE provider_id = $2 AND id = $3",
    )
    .bind(status)
    .bind(provider_id)
    .bind(id)
    .execute(pool)
    .await
    .map(|_| ())
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn delete_key_pool_entry(
    pool: &PgPool,
    provider_id: &str,
    id: &str,
) -> Result<(), ServerError> {
    sqlx::query("DELETE FROM key_pool_entries WHERE provider_id = $1 AND id = $2")
        .bind(provider_id)
        .bind(id)
        .execute(pool)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

// ── rate_limit_overrides ──────────────────────────────────────────────────────

pub async fn list_rate_limit_overrides(
    pool: &PgPool,
) -> Result<Vec<RateLimitOverrideRow>, ServerError> {
    sqlx::query_as::<_, RateLimitOverrideRow>(
        "SELECT id, rpm, tpm, created_at, updated_at \
         FROM rate_limit_overrides \
         ORDER BY id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_rate_limit_override(
    pool: &PgPool,
    id: &str,
    rpm: i32,
    tpm: i32,
) -> Result<(), ServerError> {
    sqlx::query(
        "INSERT INTO rate_limit_overrides (id, rpm, tpm, updated_at) \
         VALUES ($1, $2, $3, NOW()) \
         ON CONFLICT (id) DO UPDATE SET \
             rpm = EXCLUDED.rpm, \
             tpm = EXCLUDED.tpm, \
             updated_at = NOW()",
    )
    .bind(id)
    .bind(rpm)
    .bind(tpm)
    .execute(pool)
    .await
    .map(|_| ())
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn delete_rate_limit_override(pool: &PgPool, id: &str) -> Result<(), ServerError> {
    sqlx::query("DELETE FROM rate_limit_overrides WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

// ── config_overrides ──────────────────────────────────────────────────────────

pub async fn get_config_override(
    pool: &PgPool,
    section: &str,
) -> Result<Option<ConfigOverrideRow>, ServerError> {
    sqlx::query_as::<_, ConfigOverrideRow>(
        "SELECT section, config_json, updated_at \
         FROM config_overrides \
         WHERE section = $1",
    )
    .bind(section)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_config_override(
    pool: &PgPool,
    section: &str,
    config_json: &serde_json::Value,
) -> Result<(), ServerError> {
    sqlx::query(
        "INSERT INTO config_overrides (section, config_json, updated_at) \
         VALUES ($1, $2, NOW()) \
         ON CONFLICT (section) DO UPDATE SET \
             config_json = EXCLUDED.config_json, \
             updated_at = NOW()",
    )
    .bind(section)
    .bind(config_json)
    .execute(pool)
    .await
    .map(|_| ())
    .map_err(|e| ServerError::Database(e.to_string()))
}
