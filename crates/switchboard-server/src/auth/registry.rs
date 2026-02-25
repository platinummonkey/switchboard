//! `AuthRegistry` — tries multiple `ClientAuthValidator` implementations in
//! order, returning the first successful result.
//!
//! This allows mixed authentication schemes (e.g., static keys for CI bots,
//! JWT for developer tools) within a single deployment.

use std::sync::Arc;

use tracing::instrument;

use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

/// Holds an ordered list of [`ClientAuthValidator`] implementations.
///
/// On each incoming request, [`AuthRegistry::validate`] walks the list and
/// returns the first `Ok(ValidatedClient)`.  If all validators fail, the
/// error from the **last** validator is returned.
///
/// Construct via [`AuthRegistry::builder`] or [`AuthRegistry::new`].
pub struct AuthRegistry {
    validators: Vec<Arc<dyn ClientAuthValidator>>,
}

impl std::fmt::Debug for AuthRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthRegistry")
            .field("validators_count", &self.validators.len())
            .finish()
    }
}

impl AuthRegistry {
    /// Create a registry with an explicit list of validators.
    ///
    /// Validators are tried in the order given.
    pub fn new(validators: Vec<Arc<dyn ClientAuthValidator>>) -> Self {
        Self { validators }
    }

    /// Returns a [`AuthRegistryBuilder`] for ergonomic construction.
    pub fn builder() -> AuthRegistryBuilder {
        AuthRegistryBuilder::default()
    }

    /// Number of validators in the registry.
    pub fn len(&self) -> usize {
        self.validators.len()
    }

    /// `true` if the registry has no validators configured.
    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }

    /// Try each validator in order. Returns the first `Ok(ValidatedClient)`,
    /// or an `Err` wrapping the last failure if all validators reject.
    #[instrument(skip(self, authorization_header_value))]
    pub async fn validate(
        &self,
        authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError> {
        if self.validators.is_empty() {
            return Err(AuthError::Config("no auth validators configured".into()));
        }

        let mut last_err: Option<AuthError> = None;

        for validator in &self.validators {
            match validator.validate(authorization_header_value).await {
                Ok(client) => {
                    tracing::debug!(validator = validator.name(), "auth succeeded");
                    return Ok(client);
                }
                Err(e) => {
                    tracing::debug!(
                        validator = validator.name(),
                        error = %e,
                        "auth validator rejected request"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| AuthError::Unauthorized("authentication failed".into())))
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Ergonomic builder for [`AuthRegistry`].
#[derive(Default)]
pub struct AuthRegistryBuilder {
    validators: Vec<Arc<dyn ClientAuthValidator>>,
}

impl AuthRegistryBuilder {
    /// Append a validator to the end of the chain.
    pub fn add(mut self, validator: impl ClientAuthValidator + 'static) -> Self {
        self.validators.push(Arc::new(validator));
        self
    }

    /// Append an already-`Arc`-wrapped validator.
    pub fn add_arc(mut self, validator: Arc<dyn ClientAuthValidator>) -> Self {
        self.validators.push(validator);
        self
    }

    /// Build the registry.
    pub fn build(self) -> AuthRegistry {
        AuthRegistry {
            validators: self.validators,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::auth::validator::AuthError;

    /// A validator that always succeeds and records how many times it was called.
    struct CountingPass {
        name: &'static str,
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ClientAuthValidator for CountingPass {
        fn name(&self) -> &str {
            self.name
        }

        async fn validate(&self, _h: &str) -> Result<ValidatedClient, AuthError> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(ValidatedClient::from_static_key())
        }
    }

    /// A validator that always fails with Unauthorized.
    struct AlwaysFail {
        name: &'static str,
    }

    #[async_trait]
    impl ClientAuthValidator for AlwaysFail {
        fn name(&self) -> &str {
            self.name
        }

        async fn validate(&self, _h: &str) -> Result<ValidatedClient, AuthError> {
            Err(AuthError::Unauthorized("nope".into()))
        }
    }

    /// A validator that returns Config error.
    struct ConfigError;

    #[async_trait]
    impl ClientAuthValidator for ConfigError {
        fn name(&self) -> &str {
            "config-error"
        }

        async fn validate(&self, _h: &str) -> Result<ValidatedClient, AuthError> {
            Err(AuthError::Config("misconfigured".into()))
        }
    }

    #[tokio::test]
    async fn test_empty_registry_returns_config_error() {
        let reg = AuthRegistry::new(vec![]);
        let err = reg.validate("Bearer sk-any").await.unwrap_err();
        assert!(matches!(err, AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_single_pass_validator_succeeds() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = AuthRegistry::builder()
            .add(CountingPass {
                name: "pass1",
                count: Arc::clone(&counter),
            })
            .build();
        let result = reg.validate("Bearer anything").await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_single_fail_validator_returns_err() {
        let reg = AuthRegistry::builder()
            .add(AlwaysFail { name: "fail1" })
            .build();
        let result = reg.validate("Bearer sk-bad").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_first_pass_wins_second_not_called() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = AuthRegistry::builder()
            .add(CountingPass {
                name: "pass1",
                count: Arc::clone(&counter),
            })
            .add(CountingPass {
                name: "pass2",
                count: Arc::clone(&counter),
            })
            .build();
        reg.validate("Bearer tok").await.unwrap();
        // Only the first validator should have been called.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_fail_then_pass() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = AuthRegistry::builder()
            .add(AlwaysFail { name: "fail" })
            .add(CountingPass {
                name: "pass",
                count: Arc::clone(&counter),
            })
            .build();
        let result = reg.validate("Bearer tok").await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_all_fail_returns_last_error() {
        let reg = AuthRegistry::builder()
            .add(AlwaysFail { name: "f1" })
            .add(ConfigError)
            .build();
        let err = reg.validate("Bearer tok").await.unwrap_err();
        // Last error was ConfigError
        assert!(matches!(err, AuthError::Config(_)));
    }

    #[test]
    fn test_len_and_is_empty() {
        let empty = AuthRegistry::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let one = AuthRegistry::builder()
            .add(AlwaysFail { name: "f1" })
            .build();
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    #[test]
    fn test_add_arc() {
        let validator: Arc<dyn ClientAuthValidator> = Arc::new(AlwaysFail { name: "f1" });
        let reg = AuthRegistry::builder().add_arc(validator).build();
        assert_eq!(reg.len(), 1);
    }
}
