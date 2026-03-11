//! Typed query functions for the three persistence tables.
//!
//! All queries use diesel-async with compile-time–checked query building.

use diesel::prelude::*;
use diesel::{AsExpression, FromSqlRow};
use diesel_async::AsyncPgConnection;
use diesel_async::RunQueryDsl;

use crate::db::schema;
use crate::error::ServerError;

// ── JSONB newtype ─────────────────────────────────────────────────────────────

/// Newtype wrapper that teaches Diesel how to serialize/deserialize `JSONB`
/// columns to/from `serde_json::Value`.
///
/// Postgres JSONB wire format has a 1-byte version prefix (always `0x01`)
/// before the JSON payload.
#[derive(Debug, Clone, AsExpression, FromSqlRow)]
#[diesel(sql_type = diesel::pg::sql_types::Jsonb)]
pub struct JsonbValue(pub serde_json::Value);

// SAFETY: all pointer / slice operations in `from_sql` and `to_sql` are bounds
// checked — the 1-byte version prefix is validated with a length assertion and
// a byte-value assertion before the JSON body is parsed. `to_sql` prepends a
// fixed literal `0x01` version byte and then delegates to serde_json. No raw
// pointer arithmetic is performed.
impl diesel::deserialize::FromSql<diesel::pg::sql_types::Jsonb, diesel::pg::Pg> for JsonbValue {
    fn from_sql(bytes: diesel::pg::PgValue<'_>) -> diesel::deserialize::Result<Self> {
        let bytes = bytes.as_bytes();
        if bytes.is_empty() {
            return Err("received empty JSONB value".into());
        }
        // The first byte is the JSONB version; must be 1.
        if bytes[0] != 1 {
            return Err(format!("unsupported JSONB version: {}", bytes[0]).into());
        }
        let value = serde_json::from_slice(&bytes[1..])?;
        Ok(JsonbValue(value))
    }
}

impl diesel::serialize::ToSql<diesel::pg::sql_types::Jsonb, diesel::pg::Pg> for JsonbValue {
    fn to_sql<'b>(
        &'b self,
        out: &mut diesel::serialize::Output<'b, '_, diesel::pg::Pg>,
    ) -> diesel::serialize::Result {
        use std::io::Write;
        // Version prefix byte for JSONB.
        out.write_all(&[1])?;
        serde_json::to_writer(out, &self.0)?;
        Ok(diesel::serialize::IsNull::No)
    }
}

// ── Row types ──────────────────────────────────────────────────────────────────

/// A row from the `key_pool_entries` table.
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = schema::key_pool_entries)]
pub struct KeyPoolEntryRow {
    pub provider_id: String,
    pub id: String,
    pub key_type: String,
    pub weight: f64,
    pub status: String,
    pub source_config: JsonbValue,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl KeyPoolEntryRow {
    /// Convenience accessor — returns the `source_config` as a plain
    /// `serde_json::Value` without consuming the row.
    pub fn source_config_value(&self) -> &serde_json::Value {
        &self.source_config.0
    }
}

/// A row from the `rate_limit_overrides` table.
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = schema::rate_limit_overrides)]
pub struct RateLimitOverrideRow {
    pub id: String,
    pub rpm: i32,
    pub tpm: i32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// A row from the `config_overrides` table.
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = schema::config_overrides)]
pub struct ConfigOverrideRow {
    pub section: String,
    pub config_json: JsonbValue,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl ConfigOverrideRow {
    /// Convenience accessor — returns the `config_json` as a plain
    /// `serde_json::Value` without consuming the row.
    pub fn config_json_value(&self) -> &serde_json::Value {
        &self.config_json.0
    }
}

// ── key_pool_entries ──────────────────────────────────────────────────────────

pub async fn list_key_pool_entries(
    conn: &mut AsyncPgConnection,
) -> Result<Vec<KeyPoolEntryRow>, ServerError> {
    use schema::key_pool_entries::dsl::*;
    key_pool_entries
        .select(KeyPoolEntryRow::as_select())
        .order_by((provider_id.asc(), id.asc()))
        .load(conn)
        .await
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_key_pool_entry(
    conn: &mut AsyncPgConnection,
    p_provider_id: &str,
    p_id: &str,
    p_key_type: &str,
    p_weight: f64,
    p_status: &str,
    p_source_config: &serde_json::Value,
) -> Result<(), ServerError> {
    use diesel::upsert::excluded;
    use schema::key_pool_entries::dsl::*;

    let now = chrono::Utc::now();
    diesel::insert_into(key_pool_entries)
        .values((
            provider_id.eq(p_provider_id),
            id.eq(p_id),
            key_type.eq(p_key_type),
            weight.eq(p_weight),
            status.eq(p_status),
            source_config.eq(JsonbValue(p_source_config.clone())),
            created_at.eq(now),
            updated_at.eq(now),
        ))
        .on_conflict((provider_id, id))
        .do_update()
        .set((
            key_type.eq(excluded(key_type)),
            weight.eq(excluded(weight)),
            status.eq(excluded(status)),
            source_config.eq(excluded(source_config)),
            updated_at.eq(excluded(updated_at)),
        ))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn update_key_weight(
    conn: &mut AsyncPgConnection,
    p_provider_id: &str,
    p_id: &str,
    p_weight: f64,
) -> Result<(), ServerError> {
    use schema::key_pool_entries::dsl::*;
    diesel::update(key_pool_entries.filter(provider_id.eq(p_provider_id).and(id.eq(p_id))))
        .set((weight.eq(p_weight), updated_at.eq(chrono::Utc::now())))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn update_key_status(
    conn: &mut AsyncPgConnection,
    p_provider_id: &str,
    p_id: &str,
    p_status: &str,
) -> Result<(), ServerError> {
    use schema::key_pool_entries::dsl::*;
    diesel::update(key_pool_entries.filter(provider_id.eq(p_provider_id).and(id.eq(p_id))))
        .set((status.eq(p_status), updated_at.eq(chrono::Utc::now())))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn delete_key_pool_entry(
    conn: &mut AsyncPgConnection,
    p_provider_id: &str,
    p_id: &str,
) -> Result<(), ServerError> {
    use schema::key_pool_entries::dsl::*;
    diesel::delete(key_pool_entries.filter(provider_id.eq(p_provider_id).and(id.eq(p_id))))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

// ── rate_limit_overrides ──────────────────────────────────────────────────────

pub async fn list_rate_limit_overrides(
    conn: &mut AsyncPgConnection,
) -> Result<Vec<RateLimitOverrideRow>, ServerError> {
    use schema::rate_limit_overrides::dsl::*;
    rate_limit_overrides
        .select(RateLimitOverrideRow::as_select())
        .order_by(id.asc())
        .load(conn)
        .await
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_rate_limit_override(
    conn: &mut AsyncPgConnection,
    p_id: &str,
    p_rpm: i32,
    p_tpm: i32,
) -> Result<(), ServerError> {
    use diesel::upsert::excluded;
    use schema::rate_limit_overrides::dsl::*;

    let now = chrono::Utc::now();
    diesel::insert_into(rate_limit_overrides)
        .values((
            id.eq(p_id),
            rpm.eq(p_rpm),
            tpm.eq(p_tpm),
            created_at.eq(now),
            updated_at.eq(now),
        ))
        .on_conflict(id)
        .do_update()
        .set((
            rpm.eq(excluded(rpm)),
            tpm.eq(excluded(tpm)),
            updated_at.eq(excluded(updated_at)),
        ))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn delete_rate_limit_override(
    conn: &mut AsyncPgConnection,
    p_id: &str,
) -> Result<(), ServerError> {
    use schema::rate_limit_overrides::dsl::*;
    diesel::delete(rate_limit_overrides.filter(id.eq(p_id)))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}

// ── config_overrides ──────────────────────────────────────────────────────────

pub async fn get_config_override(
    conn: &mut AsyncPgConnection,
    p_section: &str,
) -> Result<Option<ConfigOverrideRow>, ServerError> {
    use schema::config_overrides::dsl::*;
    config_overrides
        .select(ConfigOverrideRow::as_select())
        .filter(section.eq(p_section))
        .first(conn)
        .await
        .optional()
        .map_err(|e| ServerError::Database(e.to_string()))
}

pub async fn upsert_config_override(
    conn: &mut AsyncPgConnection,
    p_section: &str,
    p_config_json: &serde_json::Value,
) -> Result<(), ServerError> {
    use diesel::upsert::excluded;
    use schema::config_overrides::dsl::*;

    let now = chrono::Utc::now();
    diesel::insert_into(config_overrides)
        .values((
            section.eq(p_section),
            config_json.eq(JsonbValue(p_config_json.clone())),
            updated_at.eq(now),
        ))
        .on_conflict(section)
        .do_update()
        .set((
            config_json.eq(excluded(config_json)),
            updated_at.eq(excluded(updated_at)),
        ))
        .execute(conn)
        .await
        .map(|_| ())
        .map_err(|e| ServerError::Database(e.to_string()))
}
