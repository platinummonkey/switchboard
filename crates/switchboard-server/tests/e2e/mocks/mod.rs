//! Mock server templates for upstream LLM providers.
//!
//! Each submodule provides wiremock [`Mock`] factories for one provider.
//! Mocks match on HTTP method and path only — no auth header matching —
//! so tests do not need real API credentials.

pub mod anthropic;
pub mod bedrock;
pub mod ollama;
pub mod openai;
pub mod vertex;
