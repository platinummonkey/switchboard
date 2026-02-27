//! Admin REST API handlers.
//!
//! All routes are under `/admin/api/v1/` and return JSON.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::admin::AdminState;
use crate::admin::auth::AdminAuth;
use crate::config::provider::KeyEntry;
use crate::config::{GuardrailsConfig, ModelSelectionConfig, RateLimitConfig, RoutingConfig};

// ── Provider / Key Pool response types ───────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ProviderSummary {
    pub id: String,
    pub api_format: String,
    pub models: Vec<String>,
    pub total_keys: usize,
    pub eligible_keys: usize,
}

#[derive(Debug, Serialize)]
pub struct KeySummary {
    pub id: String,
    pub weight: f64,
    pub source: String,
    pub status: String,
    pub total_requests: u64,
    pub errors_last_5m: u64,
    pub rate_limit_hits_last_5m: u64,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct KeyHealthResponse {
    pub id: String,
    pub status: String,
    pub total_requests: u64,
    pub errors_last_5m: u64,
    pub rate_limit_hits_last_5m: u64,
    pub avg_latency_ms: f64,
    pub last_used_secs_ago: Option<f64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateKeyRequest {
    pub weight: Option<f64>,
    pub status: Option<String>,
}

// ── Provider/Key handlers ─────────────────────────────────────────────────────

/// `GET /admin/api/v1/providers` — list all providers with key pool summary.
pub async fn list_providers(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    let summaries: Vec<ProviderSummary> = config
        .providers
        .iter()
        .map(|(id, provider_cfg)| {
            let (total, eligible) = state
                .key_pools
                .get(id)
                .map(|pool| {
                    let guard = pool.read().unwrap();
                    (guard.len(), guard.eligible_count())
                })
                .unwrap_or((0, 0));
            ProviderSummary {
                id: id.clone(),
                api_format: provider_cfg.api_format.clone(),
                models: provider_cfg.models.clone(),
                total_keys: total,
                eligible_keys: eligible,
            }
        })
        .collect();

    (
        StatusCode::OK,
        Json(serde_json::to_value(summaries).unwrap_or_default()),
    )
}

/// `GET /admin/api/v1/providers/{id}/keys` — list keys with health for a provider.
pub async fn list_provider_keys(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Path(provider_id): Path<String>,
) -> impl IntoResponse {
    let Some(pool) = state.key_pools.get(&provider_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "provider not found"})),
        );
    };

    let guard = pool.read().unwrap();
    let keys: Vec<KeySummary> = guard
        .keys
        .iter()
        .map(|k| KeySummary {
            id: k.id.clone(),
            weight: k.weight,
            source: k.source.to_string(),
            status: k.health.status.to_string(),
            total_requests: k.health.total_requests,
            errors_last_5m: k.health.errors_last_5m,
            rate_limit_hits_last_5m: k.health.rate_limit_hits_last_5m,
            avg_latency_ms: k.health.avg_latency_ms,
        })
        .collect();
    drop(guard);

    (
        StatusCode::OK,
        Json(serde_json::to_value(keys).unwrap_or_default()),
    )
}

/// `POST /admin/api/v1/providers/{id}/keys` — add a key to a provider's pool.
pub async fn add_provider_key(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Path(provider_id): Path<String>,
    Json(entry): Json<KeyEntry>,
) -> impl IntoResponse {
    // Update the in-memory hot config to persist across config reads.
    state.hot_config.update(|cfg| {
        if let Some(provider_cfg) = cfg.providers.get_mut(&provider_id) {
            provider_cfg.key_pool.keys.retain(|k| k.id != entry.id);
            provider_cfg.key_pool.keys.push(entry.clone());
        }
    });

    let Some(pool) = state.key_pools.get(&provider_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "provider not found"})),
        );
    };

    // Build a PooledKey from the entry.
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::provider::{KeySource, PooledKey};

    let config = state.hot_config.load();
    let provider_cfg = match config.providers.get(&provider_id) {
        Some(cfg) => cfg,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "provider not found in config"})),
            );
        }
    };

    let api_key = entry.api_key.as_deref().unwrap_or("");
    let (header_name, header_value_str) = match provider_cfg.api_format.as_str() {
        "anthropic" => (
            http::HeaderName::from_static("x-api-key"),
            api_key.to_string(),
        ),
        "bedrock" => (
            http::HeaderName::from_static("x-switchboard-bedrock-creds"),
            api_key.to_string(),
        ),
        _ => (
            http::HeaderName::from_static("authorization"),
            format!("Bearer {api_key}"),
        ),
    };

    let header_value = http::HeaderValue::from_str(&header_value_str)
        .unwrap_or_else(|_| http::HeaderValue::from_static("invalid"));

    let creds = UpstreamCredentials {
        header_name,
        header_value,
        expires_at: None,
    };

    let key_id = entry.id.clone();
    let pooled_key = PooledKey {
        id: entry.id.clone(),
        credentials: creds,
        weight: entry.weight,
        source: KeySource::AdminApi,
        health: crate::key_pool::KeyHealth::default(),
    };

    pool.write().unwrap().add_key(pooled_key);

    (
        StatusCode::CREATED,
        Json(serde_json::json!({"status": "added", "id": key_id})),
    )
}

/// `DELETE /admin/api/v1/providers/{id}/keys/{kid}` — remove a key.
pub async fn delete_provider_key(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Path((provider_id, key_id)): Path<(String, String)>,
) -> impl IntoResponse {
    let Some(pool) = state.key_pools.get(&provider_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "provider not found"})),
        );
    };

    // Remove from config.
    state.hot_config.update(|cfg| {
        if let Some(provider_cfg) = cfg.providers.get_mut(&provider_id) {
            provider_cfg.key_pool.keys.retain(|k| k.id != key_id);
        }
    });

    let removed = pool.write().unwrap().remove_key(&key_id);

    if removed {
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "removed", "id": key_id})),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "key not found"})),
        )
    }
}

/// `PUT /admin/api/v1/providers/{id}/keys/{kid}` — update key weight/status.
pub async fn update_provider_key(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Path((provider_id, key_id)): Path<(String, String)>,
    Json(update): Json<UpdateKeyRequest>,
) -> impl IntoResponse {
    let Some(pool) = state.key_pools.get(&provider_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "provider not found"})),
        );
    };

    let mut pool_guard = pool.write().unwrap();
    let Some(key) = pool_guard.get_key_mut(&key_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "key not found"})),
        );
    };

    if let Some(weight) = update.weight {
        key.weight = weight;
    }

    if let Some(status_str) = &update.status {
        use crate::key_pool::KeyStatus;
        key.health.status = match status_str.as_str() {
            "healthy" => KeyStatus::Healthy,
            "degraded" => KeyStatus::Degraded,
            "rate_limited" => KeyStatus::RateLimited,
            "disabled" => KeyStatus::Disabled,
            other => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("unknown status: {other}")})),
                );
            }
        };
    }
    drop(pool_guard);

    // Update the config weight if provided.
    if let Some(weight) = update.weight {
        state.hot_config.update(|cfg| {
            if let Some(provider_cfg) = cfg.providers.get_mut(&provider_id) {
                if let Some(entry) = provider_cfg
                    .key_pool
                    .keys
                    .iter_mut()
                    .find(|k| k.id == key_id)
                {
                    entry.weight = weight;
                }
            }
        });
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated", "id": key_id})),
    )
}

/// `GET /admin/api/v1/providers/{id}/keys/{kid}/health` — key health metrics.
pub async fn get_key_health(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Path((provider_id, key_id)): Path<(String, String)>,
) -> impl IntoResponse {
    let Some(pool) = state.key_pools.get(&provider_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "provider not found"})),
        );
    };

    let guard = pool.read().unwrap();
    let Some(key) = guard.keys.iter().find(|k| k.id == key_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "key not found"})),
        );
    };

    let last_used_secs = key.health.last_used.map(|t| t.elapsed().as_secs_f64());
    let last_error = key.health.last_error.as_ref().map(|(_, msg)| msg.clone());

    let resp = KeyHealthResponse {
        id: key.id.clone(),
        status: key.health.status.to_string(),
        total_requests: key.health.total_requests,
        errors_last_5m: key.health.errors_last_5m,
        rate_limit_hits_last_5m: key.health.rate_limit_hits_last_5m,
        avg_latency_ms: key.health.avg_latency_ms,
        last_used_secs_ago: last_used_secs,
        last_error,
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap_or_default()),
    )
}

// ── Model Selection handlers ──────────────────────────────────────────────────

/// `GET /admin/api/v1/model-selection` — current model selection config.
pub async fn get_model_selection(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    (StatusCode::OK, Json(config.model_selection.clone()))
}

/// `PUT /admin/api/v1/model-selection` — update model selection config.
pub async fn put_model_selection(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Json(new_config): Json<ModelSelectionConfig>,
) -> impl IntoResponse {
    state.hot_config.update(|cfg| {
        cfg.model_selection = new_config.clone();
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated"})),
    )
}

// ── Guardrails handlers ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GuardrailTestRequest {
    pub engine_index: Option<usize>,
    pub input: String,
    pub phase: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GuardrailTestResponse {
    pub matched: bool,
    pub action: Option<String>,
    pub engine: Option<String>,
    pub details: String,
}

/// `GET /admin/api/v1/guardrails` — current guardrail config.
pub async fn get_guardrails(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    (StatusCode::OK, Json(config.guardrails.clone()))
}

/// `PUT /admin/api/v1/guardrails` — update guardrail config.
pub async fn put_guardrails(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Json(new_config): Json<GuardrailsConfig>,
) -> impl IntoResponse {
    state.hot_config.update(|cfg| {
        cfg.guardrails = new_config.clone();
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated"})),
    )
}

/// `POST /admin/api/v1/guardrails/test` — test a guardrail rule against sample input.
pub async fn test_guardrail(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Json(req): Json<GuardrailTestRequest>,
) -> impl IntoResponse {
    let config = state.hot_config.load();

    if !config.guardrails.enabled {
        return (
            StatusCode::OK,
            Json(
                serde_json::to_value(GuardrailTestResponse {
                    matched: false,
                    action: None,
                    engine: None,
                    details: "guardrails are disabled".into(),
                })
                .unwrap_or_default(),
            ),
        );
    }

    let engines = &config.guardrails.engines;

    // Validate engine_index if provided.
    if let Some(idx) = req.engine_index {
        if idx >= engines.len() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "engine_index out of range"})),
            );
        }
    }

    let engines_to_test: Vec<_> = match req.engine_index {
        Some(idx) => vec![&engines[idx]],
        None => engines.iter().collect(),
    };

    for engine in engines_to_test {
        if engine.engine_type == "builtin_regex" {
            for rule in &engine.rules {
                if let Ok(re) = regex::Regex::new(&rule.pattern) {
                    if re.is_match(&req.input) {
                        return (
                            StatusCode::OK,
                            Json(
                                serde_json::to_value(GuardrailTestResponse {
                                    matched: true,
                                    action: Some(rule.action.clone()),
                                    engine: Some(engine.engine_type.clone()),
                                    details: format!("matched rule '{}'", rule.name),
                                })
                                .unwrap_or_default(),
                            ),
                        );
                    }
                }
            }
        } else if engine.engine_type == "builtin_keyword" {
            for keyword in &engine.keywords {
                if req.input.contains(keyword.as_str()) {
                    return (
                        StatusCode::OK,
                        Json(
                            serde_json::to_value(GuardrailTestResponse {
                                matched: true,
                                action: engine.action.clone(),
                                engine: Some(engine.engine_type.clone()),
                                details: format!("matched keyword '{keyword}'"),
                            })
                            .unwrap_or_default(),
                        ),
                    );
                }
            }
        }
    }

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(GuardrailTestResponse {
                matched: false,
                action: None,
                engine: None,
                details: "no rules matched".into(),
            })
            .unwrap_or_default(),
        ),
    )
}

// ── Semantic Routing handlers ─────────────────────────────────────────────────

/// `GET /admin/api/v1/routing/semantic` — current semantic routing config.
pub async fn get_semantic_routing(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    (StatusCode::OK, Json(config.routing.clone()))
}

/// `PUT /admin/api/v1/routing/semantic` — update semantic routing config.
pub async fn put_semantic_routing(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Json(new_config): Json<RoutingConfig>,
) -> impl IntoResponse {
    state.hot_config.update(|cfg| {
        cfg.routing = new_config.clone();
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated"})),
    )
}

// ── Users / Usage handlers ────────────────────────────────────────────────────

/// `GET /admin/api/v1/users` — list users with usage counters.
pub async fn list_users(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let users = state.usage.by_user();
    let total = users.len();
    (
        StatusCode::OK,
        Json(serde_json::json!({"users": users, "total": total})),
    )
}

/// `GET /admin/api/v1/usage` — current usage totals by provider/model/user.
pub async fn get_usage(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let totals = state.usage.totals();
    (StatusCode::OK, Json(totals))
}

// ── Rate Limits handlers ──────────────────────────────────────────────────────

/// `GET /admin/api/v1/rate-limits` — current rate limit config.
pub async fn get_rate_limits(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    (StatusCode::OK, Json(config.rate_limit.clone()))
}

/// `PUT /admin/api/v1/rate-limits` — update rate limit config.
pub async fn put_rate_limits(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
    Json(new_config): Json<RateLimitConfig>,
) -> impl IntoResponse {
    state.hot_config.update(|cfg| {
        cfg.rate_limit = new_config.clone();
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated"})),
    )
}

// ── System handlers ───────────────────────────────────────────────────────────

/// `GET /admin/api/v1/health` — system health (no auth required).
pub async fn admin_health(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    let config = state.hot_config.load();
    let providers: HashMap<String, serde_json::Value> = config
        .providers
        .keys()
        .map(|id| {
            let (total, eligible) = state
                .key_pools
                .get(id)
                .map(|pool| {
                    let guard = pool.read().unwrap();
                    (guard.len(), guard.eligible_count())
                })
                .unwrap_or((0, 0));
            (
                id.clone(),
                serde_json::json!({
                    "total_keys": total,
                    "eligible_keys": eligible,
                }),
            )
        })
        .collect();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "providers": providers
        })),
    )
}

/// `POST /admin/api/v1/config/reload` — reload config from disk.
pub async fn reload_config(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    match state.hot_config.reload() {
        Ok(()) => {
            tracing::info!("admin triggered config reload: success");
            (
                StatusCode::OK,
                Json(serde_json::json!({"status": "reloaded"})),
            )
        }
        Err(e) => {
            tracing::warn!(error = %e, "admin triggered config reload: failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        }
    }
}

/// `GET /admin/api/v1/config` — current config with secrets redacted.
pub async fn get_config(
    State(state): State<Arc<AdminState>>,
    _auth: AdminAuth,
) -> impl IntoResponse {
    let config = state.hot_config.load();
    let mut value = serde_json::to_value(&*config).unwrap_or_default();
    redact_secrets(&mut value);
    (StatusCode::OK, Json(value))
}

/// Recursively walk a JSON value and replace secret fields with `"<redacted>"`.
pub fn redact_secrets(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                if matches!(key.as_str(), "api_key" | "static_token" | "secret") {
                    // Only redact non-null, non-empty string values.
                    if val.is_string() && !val.as_str().map(|s| s.is_empty()).unwrap_or(true) {
                        *val = serde_json::Value::String("<redacted>".into());
                    }
                } else if key == "keys" {
                    // Redact the `keys` field in ValidatorConfig (array of strings = static keys).
                    if let serde_json::Value::Array(arr) = val {
                        if arr.iter().all(|v| v.is_string()) {
                            *val = serde_json::Value::Array(
                                arr.iter()
                                    .map(|_| serde_json::Value::String("<redacted>".into()))
                                    .collect(),
                            );
                        } else {
                            for item in arr.iter_mut() {
                                redact_secrets(item);
                            }
                        }
                    } else {
                        redact_secrets(val);
                    }
                } else {
                    redact_secrets(val);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                redact_secrets(item);
            }
        }
        _ => {}
    }
}

// ── Router builder ────────────────────────────────────────────────────────────

/// Build the admin API [`Router`] with all routes wired up.
pub fn admin_api_router(state: Arc<AdminState>) -> Router {
    Router::new()
        // Health (no auth required — added separately so AdminAuth is not applied).
        .route("/admin/api/v1/health", get(admin_health))
        // Providers / key pool.
        .route("/admin/api/v1/providers", get(list_providers))
        .route(
            "/admin/api/v1/providers/{id}/keys",
            get(list_provider_keys).post(add_provider_key),
        )
        .route(
            "/admin/api/v1/providers/{id}/keys/{kid}",
            delete(delete_provider_key).put(update_provider_key),
        )
        .route(
            "/admin/api/v1/providers/{id}/keys/{kid}/health",
            get(get_key_health),
        )
        // Model selection.
        .route(
            "/admin/api/v1/model-selection",
            get(get_model_selection).put(put_model_selection),
        )
        // Guardrails.
        .route(
            "/admin/api/v1/guardrails",
            get(get_guardrails).put(put_guardrails),
        )
        .route("/admin/api/v1/guardrails/test", post(test_guardrail))
        // Semantic routing.
        .route(
            "/admin/api/v1/routing/semantic",
            get(get_semantic_routing).put(put_semantic_routing),
        )
        // Users / Usage stubs.
        .route("/admin/api/v1/users", get(list_users))
        .route("/admin/api/v1/usage", get(get_usage))
        // Rate limits.
        .route(
            "/admin/api/v1/rate-limits",
            get(get_rate_limits).put(put_rate_limits),
        )
        // System.
        .route("/admin/api/v1/config/reload", post(reload_config))
        .route("/admin/api/v1/config", get(get_config))
        .with_state(state)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_secrets_api_key() {
        let mut val = serde_json::json!({
            "providers": {
                "anthropic": {
                    "key_pool": {
                        "keys": [{"id": "k1", "api_key": "sk-secret", "weight": 1.0}]
                    }
                }
            }
        });
        redact_secrets(&mut val);
        let key = &val["providers"]["anthropic"]["key_pool"]["keys"][0]["api_key"];
        assert_eq!(key, "<redacted>");
    }

    #[test]
    fn test_redact_secrets_static_token() {
        let mut val = serde_json::json!({
            "admin": {
                "static_token": "super-secret"
            }
        });
        redact_secrets(&mut val);
        assert_eq!(val["admin"]["static_token"], "<redacted>");
    }

    #[test]
    fn test_redact_secrets_validator_keys() {
        let mut val = serde_json::json!({
            "auth": {
                "validators": {
                    "static": {
                        "type": "static_keys",
                        "keys": ["sk-1", "sk-2"]
                    }
                }
            }
        });
        redact_secrets(&mut val);
        let keys = &val["auth"]["validators"]["static"]["keys"];
        assert_eq!(keys[0], "<redacted>");
        assert_eq!(keys[1], "<redacted>");
    }

    #[test]
    fn test_redact_secrets_leaves_non_secrets_unchanged() {
        let mut val = serde_json::json!({
            "server": {
                "listen": "0.0.0.0:8080"
            }
        });
        redact_secrets(&mut val);
        assert_eq!(val["server"]["listen"], "0.0.0.0:8080");
    }

    #[test]
    fn test_redact_secrets_null_api_key_not_redacted() {
        let mut val = serde_json::json!({
            "key_pool": {"keys": [{"id": "k1", "api_key": null}]}
        });
        redact_secrets(&mut val);
        // null is not a string, so it stays null.
        assert!(val["key_pool"]["keys"][0]["api_key"].is_null());
    }

    #[test]
    fn test_redact_secrets_empty_string_not_redacted() {
        // Empty strings don't get redacted (indicates field is absent/unset).
        let mut val = serde_json::json!({
            "admin": {"static_token": ""}
        });
        redact_secrets(&mut val);
        // Empty string — not redacted.
        assert_eq!(val["admin"]["static_token"], "");
    }
}
