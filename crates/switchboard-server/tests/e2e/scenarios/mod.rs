//! E2E test scenarios.
//!
//! Each submodule covers one concern.

pub mod admin_e2e_test;
pub mod anthropic_test;
pub mod auth_e2e_test;
pub mod bedrock_test;
pub mod failover_test;
pub mod guardrails_e2e_test;
pub mod identity_test;
pub mod key_selector_test;
pub mod model_selection_e2e_test;
pub mod ollama_test;
pub mod openai_test;
pub mod rate_limit_test;
pub mod request_id_test;
pub mod routing_test;
pub mod semantic_routing_e2e_test;
pub mod streaming_e2e_test;
pub mod usage_e2e_test;
pub mod vertex_test;

#[cfg(test)]
mod harness_smoke {
    use crate::harness::TestHarnessBuilder;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_server_starts_and_health_returns_ok() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let resp = harness.client.health().await;
        assert_eq!(resp.status().as_u16(), 200);
    }
}
