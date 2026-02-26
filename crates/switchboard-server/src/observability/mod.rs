//! OpenTelemetry observability initialisation and exports for switchboard-server.
//!
//! # Usage (in `main.rs`)
//! ```rust,ignore
//! let _otel_guard = observability::init_tracing(&config.observability)?;
//! // _otel_guard must live for the duration of the program so that spans are
//! // flushed on shutdown.
//! ```
//!
//! # Design
//! - When `config.enabled` is `false` the global OTel tracer is left as the
//!   no-op default and no OTLP exporter is started.
//! - When enabled, we build an OTLP exporter (gRPC for `:4317` endpoints,
//!   HTTP otherwise), wrap it in a `BatchSpanProcessor`, and install the
//!   resulting `SdkTracerProvider` as the global provider.
//! - `tracing-opentelemetry` bridges `tracing` spans to OTel automatically
//!   once the global provider is set.

pub mod dd_llm_obs;
pub mod metrics;
pub mod spans;

pub use spans::{ProxySpan, SpanAttributes};

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider};

use crate::config::ObservabilityConfig;
use crate::config::duration;
use crate::error::ServerError;

// ── Guard ─────────────────────────────────────────────────────────────────────

/// Returned by [`init_tracing`].
///
/// Holds the `SdkTracerProvider` (if OTel is enabled) and shuts it down — which
/// flushes all pending spans — when dropped.
pub struct OtelGuard {
    /// `Some` when OTel is enabled, `None` when disabled (no-op).
    provider: Option<SdkTracerProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(ref provider) = self.provider {
            if let Err(e) = provider.shutdown() {
                // We are in a Drop, so we cannot return an error.  Log at warn.
                tracing::warn!(error = %e, "OTel provider shutdown failed");
            }
        }
    }
}

// ── init_tracing ──────────────────────────────────────────────────────────────

/// Initialise the global OTel SDK from `config`.
///
/// Returns an [`OtelGuard`] that flushes pending spans when dropped.
///
/// # Disabled mode
/// When `config.enabled` is `false` this is a fast no-op: the global tracer
/// stays as the built-in no-op provider, and the returned guard holds
/// `None`.
///
/// # Enabled mode
/// 1. Parse `config.batch_flush_interval` using [`duration::parse`].
/// 2. Build an OTLP span exporter:
///    - gRPC (`SpanExporter::builder().with_tonic()`) when the endpoint ends
///      with `:4317`.
///    - HTTP (`SpanExporter::builder().with_http()`) otherwise.
/// 3. Wrap the exporter in a `BatchSpanProcessor` with the configured flush
///    interval.
/// 4. Build a `SdkTracerProvider` with the processor and service resource
///    attributes (`service.name`, `deployment.environment`).
/// 5. Install the provider as the global via
///    `opentelemetry::global::set_tracer_provider`.
/// 6. Return the guard.
pub fn init_tracing(config: &ObservabilityConfig) -> Result<OtelGuard, ServerError> {
    if !config.enabled {
        tracing::debug!("OTel tracing disabled; using no-op provider");
        return Ok(OtelGuard { provider: None });
    }

    // 1. Parse batch flush interval.
    let flush_interval = duration::parse(&config.batch_flush_interval)
        .map_err(|e| ServerError::Config(format!("batch_flush_interval: {e}")))?;

    // 2. Build OTLP span exporter.
    let exporter = build_otlp_exporter(&config.otlp_endpoint)
        .map_err(|e| ServerError::Config(format!("OTLP exporter build failed: {e}")))?;

    // 3. BatchSpanProcessor with the configured flush interval.
    let batch_config = BatchConfigBuilder::default()
        .with_scheduled_delay(flush_interval)
        .build();
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(batch_config)
        .build();

    // 4. Resource attributes.
    let resource = Resource::builder_empty()
        .with_service_name(config.service_name.clone())
        .with_attribute(opentelemetry::KeyValue::new(
            "deployment.environment",
            config.environment.clone(),
        ))
        .build();

    // 5. Build and install the provider.
    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .with_resource(resource)
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());

    tracing::info!(
        endpoint = %config.otlp_endpoint,
        service_name = %config.service_name,
        environment = %config.environment,
        "OTel tracing initialised"
    );

    Ok(OtelGuard {
        provider: Some(provider),
    })
}

// ── OTel layer for subscriber builder ────────────────────────────────────────

/// Build a [`tracing_opentelemetry::OpenTelemetryLayer`] that can be added to
/// a `tracing_subscriber` builder.
///
/// Returns a boxed layer so that the enabled and disabled code paths can return
/// different concrete types behind a uniform `Box<dyn Layer<S>>` interface.
///
/// Call [`init_tracing`] (and keep the returned guard alive) **before**
/// installing this layer so that the global OTel provider is configured.
///
/// # Usage
/// ```rust,ignore
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::util::SubscriberInitExt;
///
/// let _guard = observability::init_tracing(&config.observability)?;
/// tracing_subscriber::registry()
///     .with(tracing_subscriber::fmt::layer())
///     .with(observability::otel_layer(&config.observability)?)
///     .init();
/// ```
pub fn otel_layer<S>(
    config: &ObservabilityConfig,
    provider: Option<&SdkTracerProvider>,
) -> Result<Box<dyn tracing_subscriber::Layer<S> + Send + Sync + 'static>, ServerError>
where
    S: tracing::Subscriber
        + for<'span> tracing_subscriber::registry::LookupSpan<'span>
        + Send
        + Sync,
{
    if !config.enabled || provider.is_none() {
        // Return a no-op layer when OTel is disabled.
        let tracer = opentelemetry::trace::noop::NoopTracer::new();
        let layer = tracing_opentelemetry::OpenTelemetryLayer::new(tracer);
        return Ok(Box::new(layer));
    }

    // Use the SdkTracer from the provided SdkTracerProvider directly.
    // SdkTracer implements PreSampledTracer, which OpenTelemetryLayer requires.
    let tracer = provider.unwrap().tracer("switchboard-server");
    let layer = tracing_opentelemetry::OpenTelemetryLayer::new(tracer);
    Ok(Box::new(layer))
}

// ── OTLP exporter builder ────────────────────────────────────────────────────

/// Build an OTLP [`opentelemetry_otlp::SpanExporter`].
///
/// Selects gRPC transport when the endpoint ends with `:4317`, HTTP otherwise.
fn build_otlp_exporter(
    endpoint: &str,
) -> Result<opentelemetry_otlp::SpanExporter, opentelemetry_otlp::ExporterBuildError> {
    use opentelemetry_otlp::WithExportConfig as _;

    if endpoint.ends_with(":4317") {
        opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
    } else {
        opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing_disabled_returns_ok() {
        let config = ObservabilityConfig {
            enabled: false,
            ..ObservabilityConfig::default()
        };
        let result = init_tracing(&config);
        assert!(result.is_ok(), "disabled init_tracing should return Ok");
        let guard = result.unwrap();
        assert!(
            guard.provider.is_none(),
            "disabled guard should hold no provider"
        );
        // Drop the guard — should not panic.
        drop(guard);
    }

    #[test]
    fn test_otel_guard_drop_with_none_provider() {
        // A guard with no provider must drop without panicking.
        let guard = OtelGuard { provider: None };
        drop(guard);
    }
}
