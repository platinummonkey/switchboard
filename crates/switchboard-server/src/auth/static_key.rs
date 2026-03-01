//! Static API key validator.
//!
//! Validates incoming `Authorization: Bearer <key>` headers against a
//! configured list of static keys using constant-time comparison to prevent
//! timing side-channels.

use std::collections::HashMap;

use async_trait::async_trait;
use subtle::ConstantTimeEq;
use tracing::instrument;

use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

/// Validates incoming requests against a static list of pre-shared API keys.
///
/// Strips the `Bearer ` prefix (case-insensitive) from the `Authorization`
/// header value before comparison.  Uses `subtle::ConstantTimeEq` for
/// constant-time comparison to prevent timing attacks.
///
/// If a `user_map` is provided (via [`StaticKeyValidator::new_with_mappings`]),
/// the matched key is looked up in the map and the resulting `user_id` and
/// optional `team` are propagated into the returned [`ValidatedClient`].  This
/// lets per-user model overrides in `ModelSelectionConfig.overrides` fire for
/// api-key-authenticated requests without requiring an additional
/// `x-switchboard-user` header.
#[derive(Debug, Clone)]
pub struct StaticKeyValidator {
    /// Short name used in logs and metrics.
    name: String,
    /// The set of valid API keys (stored as raw bytes for CT comparison).
    keys: Vec<Vec<u8>>,
    /// Maps stripped api_key value → (user_id, optional team).
    ///
    /// Empty when constructed via [`StaticKeyValidator::new`].
    user_map: HashMap<String, (String, Option<String>)>,
}

impl StaticKeyValidator {
    /// Construct a validator from a list of plain-text API keys.
    ///
    /// No user-identity mappings are configured; the returned
    /// [`ValidatedClient`] will always have `user_id: None` for requests
    /// authenticated by this validator.
    ///
    /// # Panics
    ///
    /// Does not panic; an empty key list is permitted (though the validator
    /// will always reject).
    pub fn new(name: impl Into<String>, keys: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self {
            name: name.into(),
            keys: keys
                .into_iter()
                .map(|k| k.as_ref().as_bytes().to_vec())
                .collect(),
            user_map: HashMap::new(),
        }
    }

    /// Construct a validator with api-key → user-identity mappings.
    ///
    /// When a key is matched, the validator looks it up in `user_map` and
    /// sets `ValidatedClient.user_id` (and optionally `team` via the
    /// `__switchboard_team` claim) so that per-user policies (model overrides,
    /// rate-limit overrides, sticky key selection) fire without requiring the
    /// client to send an explicit `x-switchboard-user` header.
    ///
    /// `user_map` maps the raw API key string (without any `Bearer ` prefix)
    /// to a `(user_id, optional_team)` tuple.
    pub fn new_with_mappings(
        name: impl Into<String>,
        keys: impl IntoIterator<Item = impl AsRef<str>>,
        user_map: HashMap<String, (String, Option<String>)>,
    ) -> Self {
        Self {
            name: name.into(),
            keys: keys
                .into_iter()
                .map(|k| k.as_ref().as_bytes().to_vec())
                .collect(),
            user_map,
        }
    }

    /// Strip `Bearer ` (or `bearer `) prefix, returning the raw key string.
    fn strip_bearer(value: &str) -> &str {
        let v = value.trim();
        // Accept both `Bearer <key>` and a bare `<key>`.
        if let Some(rest) = v
            .strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
        {
            rest.trim()
        } else {
            v
        }
    }

    /// Constant-time equality check: returns `true` if `candidate` matches
    /// any configured key.
    fn matches_any(&self, candidate: &[u8]) -> bool {
        let mut found = subtle::Choice::from(0u8);
        for key in &self.keys {
            // Lengths must match for CT comparison; use a fixed-size check when
            // lengths differ to avoid leaking length information via timing.
            if key.len() == candidate.len() {
                found |= key.as_slice().ct_eq(candidate);
            }
        }
        bool::from(found)
    }
}

#[async_trait]
impl ClientAuthValidator for StaticKeyValidator {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, authorization_header_value), fields(validator = %self.name))]
    async fn validate(
        &self,
        authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError> {
        if self.keys.is_empty() {
            return Err(AuthError::Config(
                "StaticKeyValidator has no configured keys".into(),
            ));
        }

        let raw_key = Self::strip_bearer(authorization_header_value);
        if raw_key.is_empty() {
            return Err(AuthError::InvalidCredential(
                "Authorization header is empty".into(),
            ));
        }

        if self.matches_any(raw_key.as_bytes()) {
            // Look up the matched key in the user map to resolve identity.
            let (user_id, team) = self
                .user_map
                .get(raw_key)
                .map(|(uid, t)| (Some(uid.clone()), t.clone()))
                .unwrap_or((None, None));

            if let Some(ref uid) = user_id {
                tracing::debug!(
                    user_id = uid,
                    "static key validation succeeded with mapped identity"
                );
            } else {
                tracing::debug!("static key validation succeeded");
            }

            let mut client = ValidatedClient::from_static_key();
            client.user_id = user_id;
            if let Some(team_name) = team {
                client
                    .claims
                    .insert("team".to_string(), serde_json::Value::String(team_name));
            }
            Ok(client)
        } else {
            tracing::debug!("static key validation failed: key not in pool");
            Err(AuthError::Unauthorized("invalid API key".into()))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validator(keys: &[&str]) -> StaticKeyValidator {
        StaticKeyValidator::new("test", keys.iter().copied())
    }

    #[tokio::test]
    async fn test_valid_key_bearer_prefix() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("Bearer sk-secret").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_valid_key_no_prefix() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("sk-secret").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_key_rejected() {
        let v = validator(&["sk-correct"]);
        let result = v.validate("Bearer sk-wrong").await;
        assert!(result.is_err());
        matches!(result.unwrap_err(), AuthError::Unauthorized(_));
    }

    #[tokio::test]
    async fn test_empty_header_rejected() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_key_list_is_config_error() {
        let v = validator(&[]);
        let result = v.validate("Bearer sk-any").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_multiple_keys_first_valid() {
        let v = validator(&["sk-a", "sk-b", "sk-c"]);
        assert!(v.validate("Bearer sk-a").await.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_keys_last_valid() {
        let v = validator(&["sk-a", "sk-b", "sk-c"]);
        assert!(v.validate("Bearer sk-c").await.is_ok());
    }

    #[tokio::test]
    async fn test_prefix_only_no_key_rejected() {
        let v = validator(&["sk-secret"]);
        // "Bearer " with trailing whitespace and empty key
        let result = v.validate("Bearer ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_lowercase_bearer_prefix() {
        let v = validator(&["sk-lower"]);
        let result = v.validate("bearer sk-lower").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_strip_bearer_variants() {
        assert_eq!(StaticKeyValidator::strip_bearer("Bearer sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("bearer sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("  Bearer  sk-x  "), "sk-x");
    }

    #[test]
    fn test_validator_name() {
        let v = StaticKeyValidator::new("prod-keys", ["sk-1"]);
        assert_eq!(v.name(), "prod-keys");
    }

    #[tokio::test]
    async fn test_returns_static_key_client() {
        let v = validator(&["sk-test"]);
        let client = v.validate("Bearer sk-test").await.unwrap();
        // No user_map configured — user_id stays None.
        assert!(client.user_id.is_none());
        assert!(client.claims.is_empty());
    }

    // ── Tests: user_map / new_with_mappings ───────────────────────────────────

    #[tokio::test]
    async fn test_mapped_key_sets_user_id() {
        let mut user_map = HashMap::new();
        user_map.insert("sk-alice-key".to_string(), ("alice".to_string(), None));

        let v = StaticKeyValidator::new_with_mappings("test-mapped", ["sk-alice-key"], user_map);

        let client = v.validate("Bearer sk-alice-key").await.unwrap();
        assert_eq!(
            client.user_id.as_deref(),
            Some("alice"),
            "user_id must be resolved from api_key_mappings"
        );
        assert!(client.claims.is_empty(), "no team was configured");
    }

    #[tokio::test]
    async fn test_mapped_key_sets_user_id_and_team() {
        let mut user_map = HashMap::new();
        user_map.insert(
            "sk-bob-key".to_string(),
            ("bob".to_string(), Some("platform".to_string())),
        );

        let v = StaticKeyValidator::new_with_mappings("test-mapped-team", ["sk-bob-key"], user_map);

        let client = v.validate("sk-bob-key").await.unwrap();
        assert_eq!(client.user_id.as_deref(), Some("bob"));
        assert_eq!(
            client.claims.get("team").and_then(|v| v.as_str()),
            Some("platform"),
            "team claim must be set when mapping includes a team"
        );
    }

    #[tokio::test]
    async fn test_unmapped_key_has_no_user_id() {
        // Key is in the valid list but not in the user_map.
        let mut user_map = HashMap::new();
        user_map.insert("sk-alice-key".to_string(), ("alice".to_string(), None));

        let v = StaticKeyValidator::new_with_mappings(
            "test-partial-map",
            ["sk-alice-key", "sk-anon-key"],
            user_map,
        );

        // "sk-anon-key" is valid but has no mapping — user_id must be None.
        let client = v.validate("Bearer sk-anon-key").await.unwrap();
        assert!(
            client.user_id.is_none(),
            "unmapped key must produce user_id: None"
        );
        assert!(client.claims.is_empty());
    }

    #[tokio::test]
    async fn test_mapped_key_with_bearer_prefix_resolves_identity() {
        let mut user_map = HashMap::new();
        user_map.insert("sk-carol-key".to_string(), ("carol".to_string(), None));

        let v =
            StaticKeyValidator::new_with_mappings("test-bearer-strip", ["sk-carol-key"], user_map);

        // The Bearer prefix must be stripped before the user_map lookup.
        let client = v.validate("Bearer sk-carol-key").await.unwrap();
        assert_eq!(client.user_id.as_deref(), Some("carol"));
    }
}
