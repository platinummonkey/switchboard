//! Observability (OTel / DD LLM Obs) configuration.

use serde::{Deserialize, Serialize};

fn default_exporter() -> String {
    "otlp".into()
}

fn default_otlp_endpoint() -> String {
    "http://localhost:4317".into()
}

fn default_service_name() -> String {
    "switchboard".into()
}

fn default_environment() -> String {
    "production".into()
}

fn default_sample_rate() -> f64 {
    1.0
}

fn default_batch_flush_interval() -> String {
    "5s".into()
}

/// Observability export configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Export backend: "otlp" | "direct_http" (DD LLM Obs intake).
    #[serde(default = "default_exporter")]
    pub exporter: String,

    /// OTLP gRPC or HTTP endpoint (for exporter = "otlp").
    #[serde(default = "default_otlp_endpoint")]
    pub otlp_endpoint: String,

    #[serde(default = "default_service_name")]
    pub service_name: String,

    #[serde(default = "default_environment")]
    pub environment: String,

    /// Fraction of requests to trace (0.0–1.0).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: f64,

    /// Whether to include full prompt/response content in spans.
    /// Disable in production to avoid logging sensitive data.
    #[serde(default)]
    pub capture_prompts: bool,

    /// How often the OTel batch processor flushes spans to the exporter.
    #[serde(default = "default_batch_flush_interval")]
    pub batch_flush_interval: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            exporter: default_exporter(),
            otlp_endpoint: default_otlp_endpoint(),
            service_name: default_service_name(),
            environment: default_environment(),
            sample_rate: default_sample_rate(),
            capture_prompts: false,
            batch_flush_interval: default_batch_flush_interval(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full() {
        let toml = r#"
enabled = true
exporter = "otlp"
otlp_endpoint = "http://localhost:4317"
service_name = "switchboard"
environment = "production"
sample_rate = 0.5
capture_prompts = false
batch_flush_interval = "5s"
"#;
        let c: ObservabilityConfig = toml::from_str(toml).unwrap();
        assert!(c.enabled);
        assert_eq!(c.exporter, "otlp");
        assert!((c.sample_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(c.batch_flush_interval, "5s");
    }

    #[test]
    fn test_defaults() {
        let c = ObservabilityConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.exporter, "otlp");
        assert!((c.sample_rate - 1.0).abs() < f64::EPSILON);
        assert!(!c.capture_prompts);
    }
}
