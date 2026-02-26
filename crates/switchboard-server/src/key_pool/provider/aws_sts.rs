//! AWS STS `AssumeRole` key provider.
//!
//! Fetches short-lived AWS credentials by calling `sts:AssumeRole` and
//! returns them as a JSON blob suitable for the Bedrock provider.

use async_trait::async_trait;
use http::{HeaderName, HeaderValue};

use crate::auth::UpstreamCredentials;
use crate::error::ServerError;
use crate::key_pool::provider::KeyProvider;

// ── AwsStsProvider ────────────────────────────────────────────────────────────

/// Fetches temporary AWS credentials via `sts:AssumeRole`.
///
/// The returned [`UpstreamCredentials`] carry a JSON blob on the
/// `x-switchboard-bedrock-creds` header, which the Bedrock provider reads
/// to sign requests to Amazon Bedrock.
#[derive(Debug, Clone)]
pub struct AwsStsProvider {
    role_arn: String,
    region: String,
    session_name: String,
}

impl AwsStsProvider {
    /// Create a new provider.
    ///
    /// # Arguments
    /// * `role_arn` — The full ARN of the IAM role to assume.
    /// * `region`   — The AWS region used for the STS endpoint.
    pub fn new(role_arn: impl Into<String>, region: impl Into<String>) -> Self {
        let role_arn = role_arn.into();
        // Derive a session name from the role ARN (last path or colon segment),
        // trimmed to the STS 64-character limit.
        // Standard ARN format: arn:partition:service:region:account-id:resource-type/resource-id
        // We prefer the path segment after '/' if present, otherwise fall back to
        // the last ':'-delimited segment.
        let session_name = {
            let base: &str = if let Some(after_slash) = role_arn.rsplit('/').next() {
                if !after_slash.is_empty() && after_slash != role_arn.as_str() {
                    after_slash
                } else {
                    role_arn.rsplit(':').next().unwrap_or("switchboard")
                }
            } else {
                role_arn.rsplit(':').next().unwrap_or("switchboard")
            };
            base.chars().take(64).collect::<String>()
        };
        Self {
            role_arn,
            region: region.into(),
            session_name,
        }
    }
}

#[async_trait]
impl KeyProvider for AwsStsProvider {
    async fn fetch(&self) -> Result<UpstreamCredentials, ServerError> {
        tracing::debug!(
            role_arn = %self.role_arn,
            region = %self.region,
            "calling STS AssumeRole"
        );

        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_types::region::Region::new(self.region.clone()))
            .load()
            .await;

        let client = aws_sdk_sts::Client::new(&config);

        let resp = client
            .assume_role()
            .role_arn(&self.role_arn)
            .role_session_name(&self.session_name)
            .send()
            .await
            .map_err(|e| ServerError::Config(format!("STS AssumeRole failed: {e}")))?;

        let creds = resp
            .credentials()
            .ok_or_else(|| ServerError::Config("STS response missing credentials".into()))?;

        let access_key = creds.access_key_id();
        let secret_key = creds.secret_access_key();
        let session_token = creds.session_token();

        // Encode credentials as a JSON blob; the Bedrock provider reads this
        // from the `x-switchboard-bedrock-creds` header.
        let creds_json = serde_json::json!({
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": session_token,
            "region": self.region,
        })
        .to_string();

        // Convert the STS expiry (epoch seconds) to a `std::time::Instant`.
        let expires_at = {
            let exp_secs = creds.expiration().secs();
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let remaining = exp_secs.saturating_sub(now_unix);
            if remaining > 0 {
                Some(std::time::Instant::now() + std::time::Duration::from_secs(remaining as u64))
            } else {
                None
            }
        };

        let header_value = HeaderValue::from_str(&creds_json)
            .map_err(|e| ServerError::Config(format!("invalid STS credentials header: {e}")))?;

        tracing::info!(
            role_arn = %self.role_arn,
            has_expiry = expires_at.is_some(),
            "STS AssumeRole succeeded"
        );

        Ok(UpstreamCredentials {
            header_name: HeaderName::from_static("x-switchboard-bedrock-creds"),
            header_value,
            expires_at,
        })
    }

    fn description(&self) -> &str {
        &self.role_arn
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_sts_provider_new() {
        let p = AwsStsProvider::new("arn:aws:iam::123456789012:role/MyRole", "us-east-1");
        assert_eq!(p.role_arn, "arn:aws:iam::123456789012:role/MyRole");
        assert_eq!(p.region, "us-east-1");
        // Session name derived from last path segment.
        assert_eq!(p.session_name, "MyRole");
    }

    #[test]
    fn test_aws_sts_provider_description() {
        let p = AwsStsProvider::new("arn:aws:iam::123456789012:role/MyRole", "us-west-2");
        assert_eq!(p.description(), "arn:aws:iam::123456789012:role/MyRole");
    }

    #[test]
    fn test_aws_sts_provider_session_name_truncation() {
        // Session names are capped at 64 characters.
        let long_name = "A".repeat(100);
        let role_arn = format!("arn:aws:iam::123:role/{long_name}");
        let p = AwsStsProvider::new(&role_arn, "eu-west-1");
        assert_eq!(p.session_name.len(), 64);
    }

    #[test]
    fn test_aws_sts_provider_session_name_plain_arn() {
        // ARN without path separators falls back to the whole ARN (truncated).
        let p = AwsStsProvider::new("arn:aws:iam::123:role", "us-east-1");
        assert_eq!(p.session_name, "role");
    }
}
