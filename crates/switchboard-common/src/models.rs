// Model name registry: friendly names, aliases, and provider mappings.

use std::fmt;
use std::str::FromStr;

use crate::errors::SwitchboardError;

// ── Provider ──────────────────────────────────────────────────────────────────

/// Known upstream LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Bedrock,
    Vertex,
    Ollama,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
            Provider::Bedrock => "bedrock",
            Provider::Vertex => "vertex",
            Provider::Ollama => "ollama",
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Provider {
    type Err = SwitchboardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic" => Ok(Provider::Anthropic),
            "openai" => Ok(Provider::OpenAI),
            "bedrock" => Ok(Provider::Bedrock),
            "vertex" => Ok(Provider::Vertex),
            "ollama" => Ok(Provider::Ollama),
            other => Err(SwitchboardError::Internal(format!(
                "unknown provider: {other}"
            ))),
        }
    }
}

// ── Model registry ────────────────────────────────────────────────────────────

/// A single entry in the model registry.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The canonical friendly name used throughout Switchboard config and headers.
    pub name: &'static str,
    /// Alternative names this model is also known by (e.g. legacy aliases).
    pub aliases: &'static [&'static str],
    /// Which provider serves this model natively.
    pub provider: Provider,
    /// The exact model identifier the provider's API expects.
    pub provider_id: &'static str,
    /// Approximate context window size in tokens.
    pub context_window: u32,
}

/// Full model registry. All known models across all providers.
pub static MODEL_REGISTRY: &[ModelInfo] = &[
    // ── Anthropic ─────────────────────────────────────────────────────────────
    ModelInfo {
        name: "claude-sonnet-4-20250514",
        aliases: &["claude-sonnet-4", "claude-sonnet"],
        provider: Provider::Anthropic,
        provider_id: "claude-sonnet-4-20250514",
        context_window: 200_000,
    },
    ModelInfo {
        name: "claude-opus-4-20250514",
        aliases: &["claude-opus-4", "claude-opus"],
        provider: Provider::Anthropic,
        provider_id: "claude-opus-4-20250514",
        context_window: 200_000,
    },
    ModelInfo {
        name: "claude-haiku-4-5-20251001",
        aliases: &["claude-haiku-4-5", "claude-haiku"],
        provider: Provider::Anthropic,
        provider_id: "claude-haiku-4-5-20251001",
        context_window: 200_000,
    },
    // ── OpenAI ────────────────────────────────────────────────────────────────
    ModelInfo {
        name: "gpt-4o",
        aliases: &[],
        provider: Provider::OpenAI,
        provider_id: "gpt-4o",
        context_window: 128_000,
    },
    ModelInfo {
        name: "gpt-4o-mini",
        aliases: &[],
        provider: Provider::OpenAI,
        provider_id: "gpt-4o-mini",
        context_window: 128_000,
    },
    ModelInfo {
        name: "o1",
        aliases: &[],
        provider: Provider::OpenAI,
        provider_id: "o1",
        context_window: 200_000,
    },
    ModelInfo {
        name: "o3",
        aliases: &[],
        provider: Provider::OpenAI,
        provider_id: "o3",
        context_window: 200_000,
    },
    // ── Amazon Bedrock ────────────────────────────────────────────────────────
    ModelInfo {
        name: "bedrock/claude-sonnet-4-20250514",
        aliases: &[],
        provider: Provider::Bedrock,
        provider_id: "anthropic.claude-sonnet-4-20250514-v1:0",
        context_window: 200_000,
    },
    ModelInfo {
        name: "bedrock/claude-opus-4-20250514",
        aliases: &[],
        provider: Provider::Bedrock,
        provider_id: "anthropic.claude-opus-4-20250514-v1:0",
        context_window: 200_000,
    },
    ModelInfo {
        name: "bedrock/claude-haiku-4-5-20251001",
        aliases: &[],
        provider: Provider::Bedrock,
        provider_id: "anthropic.claude-haiku-4-5-20251001-v1:0",
        context_window: 200_000,
    },
    ModelInfo {
        name: "amazon.nova-pro-v1:0",
        aliases: &["nova-pro"],
        provider: Provider::Bedrock,
        provider_id: "amazon.nova-pro-v1:0",
        context_window: 300_000,
    },
    ModelInfo {
        name: "amazon.nova-lite-v1:0",
        aliases: &["nova-lite"],
        provider: Provider::Bedrock,
        provider_id: "amazon.nova-lite-v1:0",
        context_window: 300_000,
    },
    ModelInfo {
        name: "meta.llama3-3-70b-instruct-v1:0",
        aliases: &["llama3-70b"],
        provider: Provider::Bedrock,
        provider_id: "meta.llama3-3-70b-instruct-v1:0",
        context_window: 128_000,
    },
    // ── Google Vertex ─────────────────────────────────────────────────────────
    ModelInfo {
        name: "gemini-2.0-flash",
        aliases: &[],
        provider: Provider::Vertex,
        provider_id: "gemini-2.0-flash",
        context_window: 1_000_000,
    },
    ModelInfo {
        name: "gemini-2.0-pro",
        aliases: &[],
        provider: Provider::Vertex,
        provider_id: "gemini-2.0-pro",
        context_window: 2_000_000,
    },
];

// ── Lookup helpers ────────────────────────────────────────────────────────────

/// Look up a model by friendly name or alias. Returns the first match.
pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .find(|m| m.name == name || m.aliases.contains(&name))
}

/// Resolve a friendly name to the provider's native model ID.
/// Falls back to returning the name unchanged if not in the registry
/// (allows pass-through of raw provider IDs like "anthropic.claude-...").
pub fn resolve_model_id<'a>(name: &'a str, provider: &Provider) -> &'a str {
    MODEL_REGISTRY
        .iter()
        .find(|m| &m.provider == provider && (m.name == name || m.aliases.contains(&name)))
        .map(|m| m.provider_id)
        .unwrap_or(name)
}

/// Return all model names available for a given provider.
pub fn models_for_provider(provider: &Provider) -> Vec<&'static str> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| &m.provider == provider)
        .map(|m| m.name)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_anthropic_model() {
        assert_eq!(
            resolve_model_id("claude-sonnet-4-20250514", &Provider::Anthropic),
            "claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_resolve_bedrock_model() {
        assert_eq!(
            resolve_model_id("bedrock/claude-sonnet-4-20250514", &Provider::Bedrock),
            "anthropic.claude-sonnet-4-20250514-v1:0"
        );
    }

    #[test]
    fn test_resolve_unknown_model_passthrough() {
        // Unknown models pass through unchanged.
        assert_eq!(
            resolve_model_id("gpt-99-turbo", &Provider::OpenAI),
            "gpt-99-turbo"
        );
    }

    #[test]
    fn test_resolve_by_alias() {
        assert_eq!(
            resolve_model_id("claude-sonnet", &Provider::Anthropic),
            "claude-sonnet-4-20250514"
        );
        assert_eq!(
            resolve_model_id("nova-pro", &Provider::Bedrock),
            "amazon.nova-pro-v1:0"
        );
    }

    #[test]
    fn test_find_model_by_name() {
        let m = find_model("claude-opus-4-20250514").unwrap();
        assert_eq!(m.provider, Provider::Anthropic);
        assert_eq!(m.context_window, 200_000);
    }

    #[test]
    fn test_find_model_by_alias() {
        let m = find_model("claude-haiku").unwrap();
        assert_eq!(m.name, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_find_model_unknown() {
        assert!(find_model("nonexistent-model").is_none());
    }

    #[test]
    fn test_models_for_provider_anthropic() {
        let models = models_for_provider(&Provider::Anthropic);
        assert!(models.contains(&"claude-sonnet-4-20250514"));
        assert!(models.contains(&"claude-opus-4-20250514"));
        assert!(models.contains(&"claude-haiku-4-5-20251001"));
    }

    #[test]
    fn test_provider_roundtrip_from_str() {
        for p in ["anthropic", "openai", "bedrock", "vertex", "ollama"] {
            let parsed: Provider = p.parse().unwrap();
            assert_eq!(parsed.as_str(), p);
        }
    }

    #[test]
    fn test_provider_from_str_unknown() {
        assert!("unknown_provider".parse::<Provider>().is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Bedrock.to_string(), "bedrock");
    }
}
