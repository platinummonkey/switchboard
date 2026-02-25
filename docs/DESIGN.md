# Switchboard: LLM Proxy Gateway — Design Document

## Executive Summary

Switchboard is a two-component LLM proxy system written in Rust:

1. **`switchboard-server`** — A centralized gateway that routes requests to upstream LLM providers (Anthropic, OpenAI, Google, Amazon Bedrock, Ollama) with dynamic key management, semantic routing, pluggable prompt guardrails, user identity tracking, and native Datadog LLM Observability integration.

2. **`switchboard-local`** — A lightweight client-side proxy that developers run on their machines. It handles authentication to the Switchboard server, presents standard OpenAI/Anthropic-compatible endpoints to local tools (Claude Code, Cursor, OpenCode, etc.), and allows developers to plug in model preferences without touching server config.

Switchboard is **not** a multi-tenant SaaS platform. It is a single-deployment, team-oriented system with an admin-only control plane (API + UI) — no user-facing web interface.

---

## 1. Problem Statement

Teams using AI coding assistants face several operational challenges:

- **Credential sprawl**: Each developer manages their own API keys across multiple providers
- **No visibility**: No centralized view of LLM usage, costs, or quality across the team
- **No control**: No ability to enforce model policies, rate limits, routing preferences, or content guardrails
- **Provider lock-in**: Switching providers requires reconfiguring every developer's toolchain
- **No audit trail**: No record of who used what model, when, or how much it cost
- **Tool friction**: Each tool (Claude Code, Cursor, Aider) has different config formats; onboarding a new developer means configuring every tool individually
- **No content safety**: No centralized enforcement of prompt/response guardrails across tools

Existing solutions (OpenRouter, llmprovider.ai, LiteLLM) are either hosted third-party services (data leaves your network), overly complex multi-tenant platforms, or written in Python with significant overhead.

---

## 2. Competitive Landscape Analysis

### 2.1 OpenRouter

**What it is**: A hosted LLM gateway and marketplace offering access to 500+ models from 60+ providers through an OpenAI-compatible API.

**Key strengths**:
- Drop-in OpenAI API compatibility (`https://openrouter.ai/api/v1`)
- Intelligent provider routing with automatic failback (weighted by price, uptime, latency)
- Per-user tracking via `user` parameter and `session_id` grouping
- Broadcast observability — zero-instrumentation trace forwarding to Datadog, Langfuse, LangSmith, Grafana, and 10+ destinations via OTLP or webhooks
- OAuth PKCE flow for end-user-brings-their-own-credits model
- BYOK (Bring Your Own Key) — use your own provider API keys through OpenRouter's routing
- Granular provider selection: allowlist/blocklist, quantization filtering, data collection policies, ZDR (Zero Data Retention) per-request
- Auto Router (powered by NotDiamond) for prompt-aware model selection
- Management API for programmatic key provisioning with per-key credit limits and auto-reset

**Pricing model**: No markup on inference. Revenue from 5.5% fee on credit purchases. BYOK: first 1M requests/month free, then 5% of equivalent cost.

**Latency overhead**: ~25-40ms at the edge.

**Limitations for our use case**:
- Hosted third-party — all prompts transit OpenRouter's infrastructure
- No self-hosted option
- No pluggable auth — you use OpenRouter's key system
- Observability is "broadcast" (fire-and-forget), not deeply integrated
- No prompt guardrails or content safety controls
- No client-side proxy for zero-config developer onboarding

### 2.2 llmprovider.ai

**What it is**: A hosted LLM API gateway built on the open-source One API / New API ecosystem (songquanpeng/one-api). Provides unified access to multiple providers through a single API key.

**Key strengths**:
- Multi-format API compatibility: OpenAI, Anthropic (native), and Google Gemini (native) endpoints simultaneously
- Channel-based routing with weighted load balancing and automatic failover
- Secondary key distribution — abstract away provider credentials behind platform keys
- Quota-based billing with configurable group/model multipliers
- Drop-in SDK compatibility (change `baseURL` + `apiKey` on any OpenAI/Anthropic/Gemini SDK)
- Bidirectional format conversion (OpenAI ↔ Claude)

**Limitations for our use case**:
- Hosted third-party with no self-hosted offering
- Minimal observability (basic token counts and call history, no tracing)
- No distributed trace propagation
- No integration with external observability platforms
- Identity system focused on platform access control, not end-user attribution
- Limited public documentation about the hosted service
- No plugin/extension architecture
- No prompt guardrails or content safety

### 2.3 Gap Analysis — Why Switchboard

| Requirement | OpenRouter | llmprovider.ai | Switchboard |
|------------|-----------|---------------|-------------|
| Self-hosted / on-prem | No | No | **Yes** |
| Data stays in your network | No | No | **Yes** |
| Client-side proxy (zero-config tools) | No | No | **Yes** |
| Pluggable auth injection | No (fixed key system) | No | **Yes** |
| Refreshable auth state | No | No | **Yes** |
| Dynamic upstream key pool | BYOK (limited) | Channel keys | **Yes (weighted rotation)** |
| Dynamic model selection per-session | Partial (Auto Router) | No | **Yes** |
| Semantic routing | No | No | **Yes** |
| Pluggable prompt guardrails | No | No | **Yes (gRPC/HTTP callout)** |
| Native DD LLM Observability | Broadcast only | No | **Yes (deep integration)** |
| Distributed trace propagation | No | No | **Yes** |
| Sub-millisecond proxy overhead | No (~25-40ms) | Unknown | **Target: <1ms P99** |
| Rust / single binary | No | No (Go/Node) | **Yes** |
| Custom auth providers | No | No | **Yes (trait-based)** |
| Amazon Bedrock native | Via provider | No | **Yes (SigV4 + cross-region)** |
| Admin API + UI | Dashboard | Dashboard | **Yes (API-first + admin UI)** |

---

## 3. Architecture

### 3.1 High-Level Flow

```
Developer Machine                          Server Infrastructure
─────────────────                          ────────────────────

┌─────────────┐     ┌─────────────────┐     ┌────────────────────────────────────────────────────────────┐
│ Claude Code  │     │                 │     │                      switchboard-server                    │
│ Cursor       │────▶│ switchboard-    │────▶│  ┌────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │
│ OpenCode     │     │ local           │     │  │  Auth   │ │Guardrails│ │  Semantic  │ │Key Pool Mgr  │  │
│ Continue     │◀────│                 │◀────│  │  Layer  │─│  Layer   │─│  Router   │─│              │  │
│ Aider        │     │ • Auth to server│     │  └────────┘ └──────────┘ └───────────┘ └──────┬───────┘  │
└─────────────┘     │ • Model prefs   │     │                                                │          │
                    │ • Identity      │     │  ┌──────────────┐  ┌──────────────────────┐    │          │
                    └─────────────────┘     │  │ Observability │  │ Admin API + UI       │    │          │
                                            │  └──────┬───────┘  └──────────────────────┘    │          │
                                            └─────────┼──────────────────────────────────────┼──────────┘
                                                      │                                      │
                                                      ▼                                      ▼
                                               ┌──────────────┐                ┌──────────────────────┐
                                               │   Datadog     │                │   Anthropic          │
                                               │   LLM Obs     │                │   OpenAI             │
                                               └──────────────┘                │   Amazon Bedrock     │
                                                                               │   Google Vertex      │
                                                                               │   Ollama             │
                                                                               └──────────────────────┘
```

### 3.2 Component Architecture

```
switchboard/
├── crates/
│   ├── switchboard-server/            # The central gateway server
│   │   ├── src/
│   │   │   ├── main.rs                # Entry point, server bootstrap
│   │   │   ├── config/
│   │   │   │   ├── mod.rs             # Configuration types and loading
│   │   │   │   └── model_selection.rs # Model selection policies
│   │   │   ├── auth/
│   │   │   │   ├── mod.rs             # Auth trait definitions
│   │   │   │   ├── provider.rs        # AuthProvider trait (client → server)
│   │   │   │   ├── static_key.rs      # Static API key auth
│   │   │   │   ├── jwt.rs             # JWT-based auth
│   │   │   │   ├── oauth.rs           # OAuth2 token refresh
│   │   │   │   └── registry.rs        # AuthProvider registry
│   │   │   ├── key_pool/
│   │   │   │   ├── mod.rs             # Dynamic upstream key management
│   │   │   │   ├── pool.rs            # Key pool with weighted rotation
│   │   │   │   ├── health.rs          # Per-key health tracking (rate limits, errors)
│   │   │   │   └── provider.rs        # Key source trait (static, Vault, AWS STS)
│   │   │   ├── identity/
│   │   │   │   ├── mod.rs             # User identity extraction and tracking
│   │   │   │   └── resolver.rs        # Identity resolution from request context
│   │   │   ├── routing/
│   │   │   │   ├── mod.rs             # Router trait and types
│   │   │   │   ├── provider.rs        # Upstream provider definitions
│   │   │   │   ├── selector.rs        # Model/provider selection logic
│   │   │   │   ├── semantic.rs        # Semantic routing (prompt-aware model selection)
│   │   │   │   └── health.rs          # Upstream health checking
│   │   │   ├── guardrails/
│   │   │   │   ├── mod.rs             # Guardrail pipeline orchestration
│   │   │   │   ├── engine.rs          # GuardrailEngine trait
│   │   │   │   ├── builtin.rs         # Built-in rules (regex, keyword, token limits)
│   │   │   │   ├── grpc_callout.rs    # gRPC external evaluator client
│   │   │   │   ├── http_callout.rs    # HTTP external evaluator client
│   │   │   │   └── types.rs           # Verdict, Evaluation, GuardrailAction types
│   │   │   ├── proxy/
│   │   │   │   ├── mod.rs             # Core proxy logic
│   │   │   │   ├── handler.rs         # Axum request handlers
│   │   │   │   ├── stream.rs          # SSE stream forwarding and transformation
│   │   │   │   └── transform.rs       # Request/response transformation
│   │   │   ├── providers/
│   │   │   │   ├── mod.rs             # Provider abstraction
│   │   │   │   ├── openai.rs          # OpenAI-compatible provider
│   │   │   │   ├── anthropic.rs       # Anthropic Messages API provider
│   │   │   │   ├── bedrock.rs         # Amazon Bedrock (SigV4, cross-region inference)
│   │   │   │   ├── vertex.rs          # Google Vertex AI provider
│   │   │   │   └── ollama.rs          # Ollama local inference provider
│   │   │   ├── observability/
│   │   │   │   ├── mod.rs             # Observability setup
│   │   │   │   ├── dd_llm_obs.rs      # Datadog LLM Observability integration
│   │   │   │   ├── spans.rs           # GenAI span construction
│   │   │   │   └── metrics.rs         # Prometheus/StatsD metrics
│   │   │   ├── admin/
│   │   │   │   ├── mod.rs             # Admin API router
│   │   │   │   ├── api.rs             # REST API handlers
│   │   │   │   ├── ui.rs              # Embedded admin UI (static assets)
│   │   │   │   └── auth.rs            # Admin authentication (separate from proxy auth)
│   │   │   └── middleware/
│   │   │       ├── mod.rs             # Middleware composition
│   │   │       ├── auth_inject.rs     # Upstream auth injection layer
│   │   │       ├── model_override.rs  # Model selection override layer
│   │   │       ├── guardrail.rs       # Guardrail evaluation layer
│   │   │       ├── rate_limit.rs      # Per-user rate limiting
│   │   │       └── request_id.rs      # Request ID propagation
│   │   └── Cargo.toml
│   │
│   ├── switchboard-local/             # Lightweight client-side proxy
│   │   ├── src/
│   │   │   ├── main.rs               # Entry point
│   │   │   ├── config.rs             # Local config (~/.switchboard/config.toml)
│   │   │   ├── auth.rs               # Authenticate to switchboard-server
│   │   │   ├── server.rs             # Local HTTP server (binds localhost)
│   │   │   ├── proxy.rs              # Forward requests to switchboard-server
│   │   │   └── model_prefs.rs        # Local model preference injection
│   │   └── Cargo.toml
│   │
│   └── switchboard-common/            # Shared types between server and local
│       ├── src/
│       │   ├── lib.rs
│       │   ├── types.rs              # Shared request/response types
│       │   ├── protocol.rs           # Switchboard protocol headers
│       │   └── models.rs             # Model name registry and aliases
│       └── Cargo.toml
│
├── tests/
│   ├── integration/
│   │   ├── proxy_test.rs             # End-to-end proxy tests
│   │   ├── auth_test.rs              # Auth flow tests
│   │   ├── streaming_test.rs         # SSE streaming tests
│   │   ├── observability_test.rs     # DD LLM Obs integration tests
│   │   ├── model_selection_test.rs
│   │   ├── guardrails_test.rs        # Guardrail pipeline tests
│   │   ├── semantic_routing_test.rs  # Semantic routing tests
│   │   ├── key_pool_test.rs          # Dynamic key rotation tests
│   │   ├── bedrock_test.rs           # Bedrock SigV4 signing tests
│   │   ├── local_proxy_test.rs       # switchboard-local integration tests
│   │   └── admin_api_test.rs         # Admin API tests
│   ├── unit/
│   │   ├── auth/
│   │   ├── routing/
│   │   ├── transform/
│   │   ├── identity/
│   │   ├── guardrails/
│   │   └── key_pool/
│   └── fixtures/
│       ├── requests/                  # Sample request payloads
│       ├── responses/                 # Sample response payloads (incl. SSE streams)
│       └── guardrails/                # Guardrail evaluation fixtures
├── config/
│   ├── switchboard-server.toml        # Default server configuration
│   ├── switchboard-server.example.toml
│   ├── switchboard-local.toml         # Default local configuration
│   └── switchboard-local.example.toml
├── proto/
│   └── guardrails/
│       └── v1/
│           └── evaluator.proto        # Guardrail evaluator gRPC service definition
├── admin-ui/                          # Admin UI (Rust-embedded static SPA)
│   ├── index.html
│   ├── src/                           # Minimal vanilla JS or Leptos/Dioxus
│   └── dist/                          # Built assets (embedded via rust-embed)
├── Cargo.toml                         # Workspace root
├── Cargo.lock
└── docs/
    └── DESIGN.md                      # This document
```

### 3.3 Core Design Principles

1. **Zero-copy streaming**: Forward SSE bytes directly when no transformation is needed. Only parse when observability, guardrails, or transformation requires it.
2. **Trait-based extensibility**: Auth providers, key sources, model selectors, guardrail engines, and identity resolvers are traits. New implementations are plugged in without modifying core logic.
3. **Tower middleware composition**: All cross-cutting concerns (auth, guardrails, rate limiting, tracing, model override) are Tower layers, composable in any order.
4. **Fail-open by default**: If the observability pipeline is down, requests still flow. Guardrails can be configured as fail-open or fail-closed per policy.
5. **Two-binary deployment**: `switchboard-server` runs centrally, `switchboard-local` runs on developer machines. Both are single binaries with no runtime dependencies.
6. **API-first admin**: Every admin operation is available via REST API. The admin UI is a thin client over those APIs.
7. **Key pools, not key files**: Upstream credentials are managed as pools with rotation, health tracking, and dynamic provisioning — developers never touch provider API keys.

---

## 4. Client-Side Proxy: `switchboard-local`

### 4.1 Purpose

`switchboard-local` is a lightweight binary (~5MB) that runs on the developer's machine. It eliminates per-tool configuration by providing a single local endpoint that all tools connect to, and handles authentication to the central `switchboard-server`.

**Without switchboard-local** (current state):
```
Claude Code → ANTHROPIC_BASE_URL=https://switchboard.internal:8080/api
Cursor      → openai.baseUrl=https://switchboard.internal:8080/v1
Aider       → --openai-api-base https://switchboard.internal:8080/v1
             (each tool needs server URL, auth token, TLS certs, etc.)
```

**With switchboard-local**:
```
Claude Code → ANTHROPIC_BASE_URL=http://localhost:8877/api
Cursor      → openai.baseUrl=http://localhost:8877/v1
Aider       → --openai-api-base http://localhost:8877/v1
             (every tool just points to localhost, switchboard-local handles the rest)
```

### 4.2 Responsibilities

| Concern | Handled by switchboard-local |
|---------|------------------------------|
| Auth to server | Yes — authenticates via JWT, mTLS, or API key; refreshes tokens automatically |
| Tool-facing API | Yes — serves OpenAI-compatible and Anthropic-compatible endpoints on `localhost` |
| Model preferences | Yes — developer can set preferred model locally; injected as `X-Switchboard-Model` header |
| Identity | Yes — injects `X-Switchboard-User` header from local config or OS username |
| TLS termination | Yes — handles mTLS/TLS to server; tools connect over plain HTTP to localhost |
| Request buffering | No — streams requests/responses through with no buffering |
| Guardrails | No — server-side only |
| Routing decisions | No — server-side only |
| Key management | No — server-side only |

### 4.3 Local Configuration

`~/.switchboard/config.toml`:

```toml
[server]
url = "https://switchboard.internal:8080"

[auth]
# Method: "api_key" | "jwt" | "mtls" | "oauth"
method = "jwt"

[auth.jwt]
# Token can be provided directly or obtained via OAuth flow
token_command = "vault read -field=token secret/switchboard"  # shell command to get token
refresh_interval = "15m"

[auth.mtls]
cert = "~/.switchboard/client.crt"
key = "~/.switchboard/client.key"
ca = "~/.switchboard/ca.crt"

[identity]
# Auto-detected from OS user if not set
user = "cody.lee@datadog.com"
team = "platform"

[local]
listen = "127.0.0.1:8877"

[model]
# Default model preference (can be overridden per-request by the tool)
default = "claude-sonnet-4-20250514"
# Override specific model requests (tool asks for X, local sends Y)
# Useful when a tool hardcodes a model name you don't want
[model.overrides]
"gpt-4" = "claude-sonnet-4-20250514"
"gpt-4o" = "claude-sonnet-4-20250514"
```

### 4.4 Auth Flow

```
┌──────────┐        ┌───────────────────┐        ┌─────────────────────┐
│  Tool     │        │ switchboard-local │        │ switchboard-server  │
│ (Claude)  │        │                   │        │                     │
└─────┬─────┘        └────────┬──────────┘        └──────────┬──────────┘
      │ POST /v1/chat/        │                              │
      │ completions           │                              │
      │ (no auth or           │                              │
      │  dummy key)           │                              │
      │──────────────────────▶│                              │
      │                       │  POST /v1/chat/completions   │
      │                       │  Authorization: Bearer <jwt> │
      │                       │  X-Switchboard-User: cody    │
      │                       │  X-Switchboard-Model: claude │
      │                       │─────────────────────────────▶│
      │                       │                              │ (validate, route,
      │                       │                              │  guardrails, key inject,
      │                       │                              │  forward to provider)
      │                       │         SSE stream           │
      │                       │◀─────────────────────────────│
      │     SSE stream        │                              │
      │◀──────────────────────│                              │
```

### 4.5 Installation and Onboarding

```bash
# Install
brew install datadog/tap/switchboard-local   # macOS
# or
curl -sSL https://switchboard.internal/install.sh | sh

# Initialize (interactive setup — sets server URL, auth method, identity)
switchboard-local init

# Start (runs in background, auto-starts on login)
switchboard-local start

# Check status
switchboard-local status
# → Listening on 127.0.0.1:8877
# → Connected to switchboard.internal:8080
# → Auth: JWT (expires in 14m, auto-refresh enabled)
# → User: cody.lee@datadog.com (team: platform)
# → Default model: claude-sonnet-4-20250514

# Now configure any tool to point to localhost:8877 — done.
```

---

## 5. Pluggable Auth Architecture

Authentication in Switchboard operates at two distinct layers:

1. **Client → Server auth**: How `switchboard-local` (or direct clients) authenticates to `switchboard-server`
2. **Server → Upstream auth**: How `switchboard-server` authenticates to LLM providers (managed via key pools)

### 5.1 Client → Server Auth: The AuthProvider Trait

```rust
/// Represents credentials to inject into upstream requests.
pub struct UpstreamCredentials {
    /// The header name (e.g., "Authorization", "x-api-key")
    pub header_name: HeaderName,
    /// The header value (e.g., "Bearer sk-...")
    pub header_value: HeaderValue,
    /// Optional: when these credentials expire
    pub expires_at: Option<Instant>,
}

/// Trait for pluggable auth providers.
/// Each provider knows how to obtain and refresh credentials
/// for a specific upstream LLM provider.
#[async_trait]
pub trait AuthProvider: Send + Sync + 'static {
    /// Unique name for this provider (used in config and logs).
    fn name(&self) -> &str;

    /// Get current valid credentials. Implementors are responsible
    /// for caching and refreshing internally.
    async fn get_credentials(&self) -> Result<UpstreamCredentials, AuthError>;

    /// Force a credential refresh (e.g., after a 401 from upstream).
    async fn refresh(&self) -> Result<UpstreamCredentials, AuthError>;

    /// Check if credentials are still valid without fetching new ones.
    fn is_valid(&self) -> bool;
}
```

Built-in client auth validators on the server side:

| Validator | Use Case |
|-----------|----------|
| `StaticKeyValidator` | Validate against a list of known API keys |
| `JwtValidator` | Validate JWT signatures (RS256/ES256) against JWKS endpoint |
| `MtlsValidator` | Validate client certificates against trusted CA |

### 5.2 Server → Upstream Auth: Dynamic Key Pool

Rather than a single API key per provider, Switchboard manages a **pool of keys** per upstream provider. This enables:

- **Load distribution**: Spread requests across multiple keys to avoid per-key rate limits
- **Graceful degradation**: If one key hits a rate limit or is revoked, others continue serving
- **Key rotation**: Add/remove keys without downtime via the admin API
- **Per-key health tracking**: Track error rates, rate limit hits, and latency per key
- **Dynamic provisioning**: Keys can be sourced from Vault, AWS STS, or other secret managers

#### Key Pool Architecture

```rust
/// A single key in the pool with health metadata.
pub struct PooledKey {
    pub id: String,
    pub credentials: UpstreamCredentials,
    pub weight: f64,          // selection weight (0.0 - 1.0)
    pub source: KeySource,
    pub health: KeyHealth,
}

pub struct KeyHealth {
    pub total_requests: u64,
    pub errors_last_5m: u64,
    pub rate_limit_hits_last_5m: u64,
    pub avg_latency_ms: f64,
    pub last_used: Option<Instant>,
    pub last_error: Option<(Instant, String)>,
    pub status: KeyStatus,
}

pub enum KeyStatus {
    Healthy,
    Degraded,      // elevated errors, reduced weight
    RateLimited,   // temporarily deprioritized
    Disabled,      // manually or automatically disabled
}

pub enum KeySource {
    Static,                     // from config file
    Vault { path: String },     // HashiCorp Vault
    AwsSts { role_arn: String }, // AWS STS AssumeRole
    AdminApi,                   // added via admin API at runtime
}

/// Trait for key pool selection strategy.
pub trait KeySelector: Send + Sync {
    /// Select the best key from the pool for this request.
    fn select(&self, pool: &[PooledKey], request: &RequestContext) -> Option<&PooledKey>;
}
```

#### Built-in Key Selectors

| Selector | Algorithm |
|----------|-----------|
| `WeightedRandom` | Weighted random selection, respecting health scores. Default. |
| `RoundRobin` | Cycle through healthy keys in order |
| `LeastLoaded` | Pick the key with the lowest recent request count |
| `Sticky` | Same user always gets the same key (for providers with per-key context caching) |

#### Key Pool Configuration

```toml
[providers.anthropic.key_pool]
selector = "weighted_random"

[[providers.anthropic.key_pool.keys]]
id = "anthropic-prod-1"
api_key = "${ANTHROPIC_API_KEY_1}"
weight = 1.0

[[providers.anthropic.key_pool.keys]]
id = "anthropic-prod-2"
api_key = "${ANTHROPIC_API_KEY_2}"
weight = 1.0

[[providers.anthropic.key_pool.keys]]
id = "anthropic-backup"
api_key = "${ANTHROPIC_API_KEY_BACKUP}"
weight = 0.3  # lower weight, used less frequently

[providers.bedrock.key_pool]
selector = "round_robin"

[[providers.bedrock.key_pool.keys]]
id = "bedrock-us-east"
type = "aws_sts"
role_arn = "arn:aws:iam::123456789:role/switchboard-bedrock"
region = "us-east-1"
refresh_interval = "45m"  # STS tokens last 1h, refresh at 45m

[[providers.bedrock.key_pool.keys]]
id = "bedrock-us-west"
type = "aws_sts"
role_arn = "arn:aws:iam::123456789:role/switchboard-bedrock"
region = "us-west-2"
refresh_interval = "45m"
```

### 5.3 Refreshable State Machine

```
                    ┌─────────┐
                    │  Init   │
                    └────┬────┘
                         │ get_credentials()
                         ▼
                    ┌─────────┐
              ┌────▶│  Valid  │◀────┐
              │     └────┬────┘     │
              │          │ expires_at - buffer
              │          ▼          │
              │     ┌──────────┐   │
              │     │Refreshing│───┘ success
              │     └────┬─────┘
              │          │ failure (retries exhausted)
              │          ▼
              │     ┌─────────┐
              └─────│  Stale  │ (use last-known-good, log error)
                    └─────────┘
```

Credentials are refreshed proactively in the background before expiry. The refresh buffer is configurable (default: 30 seconds before expiry). If refresh fails, the last-known-good credentials are used while retries continue.

This state machine applies to both client → server auth tokens (in `switchboard-local`) and server → upstream keys (in the key pool, for STS/Vault-sourced keys).

---

## 6. User Identity

### 6.1 Identity Resolution

Switchboard extracts user identity from incoming requests using a chain of resolvers:

1. **Header-based**: `X-Switchboard-User` header (injected by `switchboard-local`)
2. **JWT claims**: Extract `sub`, `email`, or custom claim from bearer token
3. **API key mapping**: Map incoming API key to a known user in config
4. **mTLS CN**: Extract Common Name from client certificate
5. **Tool-specific**: Extract from tool-specific headers (e.g., Cursor sends user context)

The resolved identity is attached to every trace span and metric as a tag.

### 6.2 Identity Data Model

```rust
pub struct UserIdentity {
    /// Primary identifier (email, username, or opaque ID)
    pub id: String,
    /// Optional display name
    pub name: Option<String>,
    /// Optional team/group
    pub team: Option<String>,
    /// Source of identity resolution
    pub source: IdentitySource,
}

pub enum IdentitySource {
    Header,
    JwtClaim(String),  // claim name
    ApiKeyMapping,
    MtlsCn,
    ToolSpecific(String),  // tool name
    Anonymous,
}
```

---

## 7. Routing and Model Selection

### 7.1 Upstream Provider Configuration

```toml
[providers.anthropic]
base_url = "https://api.anthropic.com"
api_format = "anthropic"  # native Anthropic Messages API
models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"]
health_check_interval = "30s"
timeout = "300s"
max_concurrent = 100

[providers.openai]
base_url = "https://api.openai.com"
api_format = "openai"
models = ["gpt-4o", "gpt-4o-mini", "o1", "o3"]

[providers.bedrock]
api_format = "bedrock"
region = "us-east-1"
cross_region_inference = true  # enable cross-region inference profiles
models = [
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
]
# SigV4 auth is handled automatically via the key pool (AWS STS keys)

[providers.vertex]
api_format = "vertex"
project_id = "my-gcp-project"
region = "us-central1"
models = ["gemini-2.0-flash", "gemini-2.0-pro"]

[providers.ollama]
base_url = "http://gpu-box.internal:11434"
api_format = "openai"  # Ollama speaks OpenAI-compatible
models = ["llama3.3:70b", "codestral:latest"]
```

### 7.2 Amazon Bedrock Integration

Bedrock requires special handling due to AWS SigV4 request signing and its unique API format:

**SigV4 Signing**: Every request to Bedrock must be signed with AWS credentials. The key pool manages STS-sourced temporary credentials and the Bedrock provider module handles signing via the `aws-sigv4` crate.

**Cross-Region Inference**: Bedrock supports inference profiles that automatically route to the nearest region with capacity. Switchboard supports this via the `cross_region_inference` flag.

**Model ID Format**: Bedrock uses `provider.model-version` format (e.g., `anthropic.claude-sonnet-4-20250514-v1:0`). Switchboard maps friendly names (e.g., `claude-sonnet-4-20250514`) to Bedrock model IDs in the model selection layer.

**Request/Response Transform**: The Bedrock Converse API differs from both OpenAI and Anthropic formats. The `bedrock` provider module handles bidirectional transformation.

```rust
// Bedrock provider handles the full signing flow:
impl UpstreamProvider for BedrockProvider {
    async fn send(&self, request: ProxiedRequest, key: &PooledKey) -> Result<ProxiedResponse> {
        let aws_creds = key.as_aws_credentials()?;
        let bedrock_request = self.transform_request(request)?;
        let signed = self.sign_request(bedrock_request, &aws_creds)?;
        let response = self.client.execute(signed).await?;
        self.transform_response(response)
    }
}
```

### 7.3 Model Selection Policies

Switchboard supports four model selection modes:

**Static override**: Force a specific model regardless of what the client requests.
```toml
[model_selection]
mode = "static"
model = "claude-sonnet-4-20250514"
```

**Mapping**: Map requested models to different upstream models.
```toml
[model_selection]
mode = "mapping"

[model_selection.mappings]
"gpt-4" = "claude-sonnet-4-20250514"
"gpt-4o" = "claude-sonnet-4-20250514"
"gpt-3.5-turbo" = "claude-haiku-4-5-20251001"
```

**Dynamic**: Per-session model selection via headers or session state.
```toml
[model_selection]
mode = "dynamic"
header = "X-Switchboard-Model"  # client can request specific model
fallback = "claude-sonnet-4-20250514"  # if header not present
allowed_models = ["claude-*", "gpt-4o"]  # glob patterns for allowed models
```

**Semantic**: Route based on prompt content analysis (see section 8).
```toml
[model_selection]
mode = "semantic"
```

### 7.4 Session-Level Model Selection

Clients can set a model for an entire session via:
- **Header**: `X-Switchboard-Model: claude-opus-4-20250514` on any request
- **Local config**: `switchboard-local` injects the developer's preferred model
- **Session cookie**: Switchboard remembers the last model selection per session ID
- **Config override per user/team**: Specific users or teams can have default model overrides (set via admin API)

---

## 8. Semantic Routing

Semantic routing analyzes the content of incoming prompts to select the optimal model and/or provider. This is distinct from model selection policies — it is content-aware.

### 8.1 Routing Signals

| Signal | What it detects | Routing action |
|--------|----------------|----------------|
| **Task complexity** | Simple Q&A vs. multi-step reasoning vs. code generation | Route simple tasks to cheaper/faster models, complex tasks to capable models |
| **Language/domain** | Code (with language detection), natural language, math, creative writing | Route code tasks to code-optimized models |
| **Context length** | Token count estimation of the prompt | Route to providers with sufficient context windows |
| **Tool use** | Presence of tool definitions in the request | Route to models with strong tool use support |
| **Cost sensitivity** | User/team cost tier | Prefer cheaper providers for cost-sensitive teams |

### 8.2 Implementation

Semantic routing uses a lightweight classifier that runs on the prompt before forwarding. It does **not** call an LLM for classification — that would defeat the purpose.

```rust
#[async_trait]
pub trait SemanticClassifier: Send + Sync {
    /// Classify the prompt and return routing hints.
    async fn classify(&self, request: &ClassificationInput) -> ClassificationResult;
}

pub struct ClassificationInput {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    pub estimated_tokens: usize,
}

pub struct ClassificationResult {
    pub task_type: TaskType,
    pub complexity: Complexity,
    pub recommended_models: Vec<ModelRecommendation>,
}

pub enum TaskType {
    CodeGeneration { language: Option<String> },
    CodeReview,
    NaturalLanguage,
    Reasoning,
    CreativeWriting,
    DataAnalysis,
    ToolUse,
    Unknown,
}

pub enum Complexity {
    Simple,   // short answers, lookups, formatting
    Medium,   // moderate reasoning, standard code tasks
    Complex,  // multi-step reasoning, architecture, long generation
}
```

### 8.3 Routing Rules

Rules map classification results to model preferences:

```toml
[routing.semantic]
enabled = true
# Classifier: "heuristic" (rule-based, zero latency) | "embedding" (vector similarity)
classifier = "heuristic"

[[routing.semantic.rules]]
task_type = "code_generation"
complexity = "complex"
preferred_models = ["claude-opus-4-20250514", "o3"]
fallback_models = ["claude-sonnet-4-20250514", "gpt-4o"]

[[routing.semantic.rules]]
task_type = "code_generation"
complexity = "simple"
preferred_models = ["claude-haiku-4-5-20251001", "gpt-4o-mini"]

[[routing.semantic.rules]]
task_type = "natural_language"
complexity = "simple"
preferred_models = ["claude-haiku-4-5-20251001", "amazon.nova-lite-v1:0"]

[[routing.semantic.rules]]
task_type = "tool_use"
preferred_models = ["claude-sonnet-4-20250514", "gpt-4o"]

# Default if no rule matches
[routing.semantic.default]
preferred_models = ["claude-sonnet-4-20250514"]
```

### 8.4 Classifier Modes

**Heuristic** (default, zero latency): Rule-based classification using:
- Message count and length → complexity estimation
- Presence of code blocks/fences → code task detection
- Presence of `tools` array → tool use detection
- System prompt keyword analysis → domain detection
- Token count → context window requirements

**Embedding** (optional, ~1ms latency): Precompute embeddings for task categories. At request time, embed the first N characters of the prompt and find the nearest task category via cosine similarity. Uses a small local embedding model (no external calls).

---

## 9. Prompt Guardrails

### 9.1 Overview

Guardrails evaluate requests and/or responses against configurable policies before they reach the LLM or the user. The guardrail system is **pluggable** — built-in rules handle common cases, and external gRPC/HTTP services handle custom evaluation logic.

### 9.2 Guardrail Pipeline

```
Request arrives
    │
    ▼
┌──────────────────────────┐
│  Pre-request guardrails  │──── Block / Modify / Pass
│  (evaluate prompt)       │
└──────────┬───────────────┘
           │ Pass
           ▼
┌──────────────────────────┐
│  Forward to LLM provider │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Post-response guardrails│──── Block / Modify / Pass
│  (evaluate completion)   │
└──────────┬───────────────┘
           │ Pass
           ▼
Return response to client
```

For streaming responses, post-response guardrails operate in one of two modes:
- **Buffered**: Accumulate the full response before evaluating (adds latency but full accuracy)
- **Chunked**: Evaluate every N tokens during streaming (lower latency, may miss cross-chunk patterns)
- **Async audit**: Stream through immediately but evaluate asynchronously; log violations without blocking

### 9.3 The GuardrailEngine Trait

```rust
/// The result of a guardrail evaluation.
pub struct GuardrailVerdict {
    pub action: GuardrailAction,
    pub engine: String,        // which engine produced this verdict
    pub rule: String,          // which rule triggered
    pub reason: Option<String>,
    pub confidence: f64,       // 0.0 - 1.0
    pub latency: Duration,
}

pub enum GuardrailAction {
    /// Allow the request/response to proceed.
    Pass,
    /// Block the request/response entirely. Return the provided message to the client.
    Block { message: String },
    /// Modify the content before proceeding (e.g., redact PII).
    Modify { modified_content: String },
    /// Log for audit but allow through.
    AuditLog { severity: AuditSeverity },
}

pub enum AuditSeverity {
    Info,
    Warning,
    Critical,
}

/// Trait for pluggable guardrail engines.
#[async_trait]
pub trait GuardrailEngine: Send + Sync {
    fn name(&self) -> &str;

    /// Evaluate a request (pre-LLM).
    async fn evaluate_request(&self, request: &GuardrailInput) -> Result<GuardrailVerdict>;

    /// Evaluate a response (post-LLM).
    async fn evaluate_response(&self, response: &GuardrailInput) -> Result<GuardrailVerdict>;
}

pub struct GuardrailInput {
    pub content: String,           // the text to evaluate
    pub messages: Vec<Message>,    // full message context
    pub user: Option<UserIdentity>,
    pub model: String,
    pub metadata: HashMap<String, String>,
}
```

### 9.4 Built-in Guardrail Engines

| Engine | What it does | Latency |
|--------|-------------|---------|
| `RegexEngine` | Match against configurable regex patterns (PII patterns, banned phrases, secrets) | <1ms |
| `KeywordEngine` | Keyword/phrase blocklist and allowlist | <1ms |
| `TokenLimitEngine` | Enforce max input/output token limits per user/team | <1ms |
| `SecretDetectionEngine` | Detect API keys, passwords, private keys in prompts | <1ms |
| `TopicBlockEngine` | Block requests about specific topics (configurable keyword sets) | <1ms |

### 9.5 External Evaluator Callout

For custom guardrail logic that goes beyond pattern matching, Switchboard supports calling external services via gRPC or HTTP.

#### gRPC Evaluator

```protobuf
// proto/guardrails/v1/evaluator.proto

syntax = "proto3";
package guardrails.v1;

service GuardrailEvaluator {
  // Evaluate a prompt before it reaches the LLM.
  rpc EvaluateRequest(EvaluateRequestInput) returns (EvaluateResponse);

  // Evaluate a completion before it reaches the user.
  rpc EvaluateCompletion(EvaluateCompletionInput) returns (EvaluateResponse);
}

message EvaluateRequestInput {
  repeated Message messages = 1;
  string model = 2;
  string user_id = 3;
  string team = 4;
  map<string, string> metadata = 5;
}

message EvaluateCompletionInput {
  string completion = 1;
  repeated Message messages = 2;  // original messages for context
  string model = 3;
  string user_id = 4;
  map<string, string> metadata = 5;
}

message Message {
  string role = 1;
  string content = 2;
}

message EvaluateResponse {
  Action action = 1;
  string rule = 2;
  string reason = 3;
  double confidence = 4;
}

enum Action {
  PASS = 0;
  BLOCK = 1;
  MODIFY = 2;
  AUDIT_LOG = 3;
}
```

#### HTTP Evaluator

For teams that prefer REST over gRPC:

```
POST /evaluate/request
POST /evaluate/completion

Request body:
{
  "messages": [...],
  "model": "claude-sonnet-4-20250514",
  "user_id": "cody.lee",
  "team": "platform",
  "metadata": {}
}

Response body:
{
  "action": "pass" | "block" | "modify" | "audit_log",
  "rule": "pii-detection",
  "reason": "Found SSN pattern in prompt",
  "confidence": 0.95,
  "modified_content": "..."  // only if action == "modify"
}
```

### 9.6 Guardrail Configuration

```toml
[guardrails]
enabled = true
fail_mode = "open"  # "open" (allow on engine error) | "closed" (block on engine error)
timeout = "500ms"   # max time for guardrail evaluation

# Streaming response mode: "buffered" | "chunked" | "async_audit"
streaming_mode = "async_audit"

[[guardrails.engines]]
type = "builtin_regex"
phase = "pre_request"
rules = [
    { name = "ssn", pattern = "\\b\\d{3}-\\d{2}-\\d{4}\\b", action = "block" },
    { name = "api_key", pattern = "\\b(sk-|pk_|AKIA)[A-Za-z0-9]{20,}\\b", action = "block" },
]

[[guardrails.engines]]
type = "builtin_secret_detection"
phase = "pre_request"
action = "block"

[[guardrails.engines]]
type = "grpc"
phase = "pre_request"
endpoint = "guardrails-service.internal:50051"
tls = true
timeout = "200ms"

[[guardrails.engines]]
type = "http"
phase = "post_response"
endpoint = "https://guardrails-service.internal/evaluate/completion"
timeout = "300ms"
# Headers to pass to the external service
headers = { "Authorization" = "Bearer ${GUARDRAIL_SERVICE_TOKEN}" }
```

### 9.7 Guardrail Observability

Every guardrail evaluation emits a span:

```
[switchboard.guardrail]         kind: task
    Attributes:
      guardrail.engine = "grpc_evaluator"
      guardrail.phase = "pre_request"
      guardrail.action = "pass"
      guardrail.rule = ""
      guardrail.confidence = 1.0
      guardrail.latency_ms = 12
```

Blocked requests emit a metric: `switchboard.guardrail.blocked{engine, rule, user, team}`.

---

## 10. Datadog LLM Observability Integration

### 10.1 Integration Strategy

Switchboard uses **OpenTelemetry with GenAI Semantic Conventions** (v1.37+) as the primary integration path, with a fallback to the direct HTTP API for LLM-specific fields not yet covered by OTel conventions.

**Why OTel first**:
- Vendor-neutral instrumentation code
- Datadog officially supports OTel GenAI semantic conventions
- The `datadog-opentelemetry` crate provides Rust-native export
- Distributed trace propagation works automatically

### 10.2 Span Structure

Each proxied LLM request generates a trace with the following span hierarchy:

```
[switchboard.proxy]                kind: workflow
  ├── [switchboard.auth]           kind: task       (auth resolution)
  ├── [switchboard.guardrail.pre]  kind: task       (pre-request guardrails)
  ├── [switchboard.route]          kind: task       (model/provider selection)
  ├── [switchboard.key_select]     kind: task       (key pool selection)
  ├── [llm.chat]                   kind: llm        (the actual LLM call)
  │     Attributes:
  │       gen_ai.operation.name = "chat"
  │       gen_ai.provider.name = "anthropic"
  │       gen_ai.request.model = "claude-sonnet-4-20250514"
  │       gen_ai.response.model = "claude-sonnet-4-20250514"
  │       gen_ai.usage.input_tokens = 1500
  │       gen_ai.usage.output_tokens = 800
  │       gen_ai.request.temperature = 0.7
  │       gen_ai.response.finish_reasons = ["stop"]
  │     Metrics:
  │       input_cost = 0.0045
  │       output_cost = 0.012
  │       total_cost = 0.0165
  │       time_to_first_token = 0.34
  └── [switchboard.guardrail.post]  kind: task      (post-response guardrails)
```

### 10.3 Tags and Metadata

Every span includes:
- `env`: deployment environment
- `service`: "switchboard"
- `ml_app`: extracted from client request or config (per-application attribution)
- `user.id`: resolved user identity
- `user.team`: resolved team
- `switchboard.provider`: upstream provider name
- `switchboard.model.requested`: model the client asked for
- `switchboard.model.actual`: model actually used (after selection/mapping)
- `switchboard.model.selection_reason`: why this model was chosen (static, mapping, semantic, dynamic)
- `switchboard.tool`: client tool name (claude-code, cursor, etc.)
- `switchboard.key_pool.key_id`: which key from the pool was used (for debugging key issues)
- `switchboard.guardrail.pre_action`: pre-request guardrail action taken
- `switchboard.guardrail.post_action`: post-response guardrail action taken

### 10.4 Streaming Observability

For streaming responses, Switchboard:
1. Records `time_to_first_token` from the first SSE data chunk
2. Accumulates token counts from `usage` fields in the final SSE chunk (OpenAI format) or response headers (Anthropic format)
3. Finalizes the span only after the stream completes or errors
4. If the client disconnects mid-stream, records a partial span with `status: error` and the tokens consumed so far

### 10.5 Export Pipeline

```
Switchboard (tracing + tracing-opentelemetry)
    │
    ▼
OpenTelemetry SDK (batch span processor, 5s flush interval)
    │
    ▼
OTLP Exporter (gRPC or HTTP)
    │
    ▼
Datadog Agent (port 4317/4318, OTLP ingest enabled)
    │
    ▼
Datadog LLM Observability
```

Alternatively, for environments without a Datadog Agent:
```
Switchboard → Direct HTTP API → llmobs-intake.datadoghq.com
```

---

## 11. Tool Integration

### 11.1 With switchboard-local (recommended)

Developers install `switchboard-local` once and configure all tools to point to `localhost:8877`:

| Tool | Config |
|------|--------|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8877/api` |
| Cursor | OpenAI base URL: `http://localhost:8877/v1` |
| OpenCode | OpenAI base URL: `http://localhost:8877/v1` |
| Continue | OpenAI base URL: `http://localhost:8877/v1` |
| Aider | `--openai-api-base http://localhost:8877/v1` |

### 11.2 Direct to server (advanced)

For CI/CD pipelines, automated systems, or developers who prefer not to run a local proxy:

| Tool | Config |
|------|--------|
| Claude Code | `ANTHROPIC_BASE_URL=https://switchboard.internal:8080/api` |
| Cursor | OpenAI base URL: `https://switchboard.internal:8080/v1` |
| Any HTTP client | Standard OpenAI-compatible API at `https://switchboard.internal:8080/v1` |

### 11.3 API Surface

**OpenAI-compatible** (primary):
- `POST /v1/chat/completions` — Chat completions (streaming and non-streaming)
- `POST /v1/completions` — Legacy completions
- `POST /v1/embeddings` — Embeddings
- `GET /v1/models` — List available models (filtered by config and user permissions)

**Anthropic-compatible**:
- `POST /api/v1/messages` — Anthropic Messages API (streaming and non-streaming)

**Operational**:
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics

---

## 12. Admin API and UI

### 12.1 Design Philosophy

The admin control plane is **API-first**: every operation is available via a REST API with JSON request/response bodies. The admin UI is a thin static SPA that consumes these APIs. There is **no user-facing web interface** — developers interact exclusively through `switchboard-local` and their tools.

### 12.2 Admin Authentication

Admin access is separate from proxy auth:

```toml
[admin]
enabled = true
listen = "127.0.0.1:9090"  # separate port, bind to localhost or internal network
auth = "jwt"                # "jwt" | "static_token" | "mtls"
jwt_issuer = "https://auth.internal"
jwt_audience = "switchboard-admin"
allowed_roles = ["switchboard-admin", "platform-team"]
```

### 12.3 Admin REST API

**Key Pool Management**:
```
GET    /admin/api/v1/providers                        # List all providers and their key pools
GET    /admin/api/v1/providers/{id}/keys              # List keys in a provider's pool
POST   /admin/api/v1/providers/{id}/keys              # Add a key to the pool
DELETE /admin/api/v1/providers/{id}/keys/{key_id}     # Remove a key
PUT    /admin/api/v1/providers/{id}/keys/{key_id}     # Update key weight/status
GET    /admin/api/v1/providers/{id}/keys/{key_id}/health  # Key health metrics
```

**Model Selection**:
```
GET    /admin/api/v1/model-selection                  # Current model selection config
PUT    /admin/api/v1/model-selection                  # Update model selection policy
GET    /admin/api/v1/model-selection/overrides         # Per-user/team overrides
PUT    /admin/api/v1/model-selection/overrides/{id}   # Set override for user/team
DELETE /admin/api/v1/model-selection/overrides/{id}   # Remove override
```

**Guardrails**:
```
GET    /admin/api/v1/guardrails                       # List all guardrail engines and rules
PUT    /admin/api/v1/guardrails                       # Update guardrail config
POST   /admin/api/v1/guardrails/test                  # Test a guardrail against sample input
GET    /admin/api/v1/guardrails/audit-log             # Recent guardrail actions (block/modify/audit)
```

**Semantic Routing**:
```
GET    /admin/api/v1/routing/semantic                 # Current semantic routing rules
PUT    /admin/api/v1/routing/semantic                 # Update semantic routing rules
POST   /admin/api/v1/routing/semantic/classify        # Test classification on sample input
```

**Users and Usage**:
```
GET    /admin/api/v1/users                            # List known users with usage stats
GET    /admin/api/v1/users/{id}                       # User detail + recent requests
GET    /admin/api/v1/usage                            # Aggregate usage stats
GET    /admin/api/v1/usage/by-model                   # Usage breakdown by model
GET    /admin/api/v1/usage/by-user                    # Usage breakdown by user
GET    /admin/api/v1/usage/by-team                    # Usage breakdown by team
GET    /admin/api/v1/usage/costs                      # Cost tracking
```

**Rate Limits**:
```
GET    /admin/api/v1/rate-limits                      # Current rate limit config
PUT    /admin/api/v1/rate-limits                      # Update global rate limits
PUT    /admin/api/v1/rate-limits/overrides/{id}       # Set per-user/team override
```

**System**:
```
GET    /admin/api/v1/config                           # Current config (secrets redacted)
POST   /admin/api/v1/config/reload                    # Hot-reload config from file
GET    /admin/api/v1/health                           # Detailed health (upstream connectivity, key pool status)
```

### 12.4 Admin UI

The admin UI is a lightweight static SPA embedded in the `switchboard-server` binary via `rust-embed`. It provides:

- **Dashboard**: Overview of request volume, error rates, costs, top models, top users
- **Key Pool Manager**: View key health, add/remove/disable keys, see per-key metrics
- **Guardrail Console**: View recent evaluations, test rules against sample input, toggle engines
- **Routing Viewer**: Current routing rules, test semantic classification, view model selection stats
- **Usage Explorer**: Filter usage by user, team, model, provider, time range
- **Rate Limit Editor**: View and modify rate limits per user/team
- **Config Viewer**: Current running config (read-only, secrets redacted)

The UI is optional and adds ~2MB to the binary. It can be disabled at compile time with `--no-default-features`.

---

## 13. Configuration

### 13.1 Configuration Format

TOML is the primary configuration format, with environment variable overrides. Config can be hot-reloaded via the admin API without restarting the server.

```toml
[server]
listen = "0.0.0.0:8080"
graceful_shutdown_timeout = "30s"

[admin]
enabled = true
listen = "127.0.0.1:9090"
auth = "static_token"
static_token = "${SWITCHBOARD_ADMIN_TOKEN}"

[identity]
resolvers = ["header", "jwt", "mtls_cn", "api_key"]
header_name = "X-Switchboard-User"
jwt_claim = "email"

# ─── Client → Server Auth ───

[auth.validators.jwt]
type = "jwt"
jwks_url = "https://auth.internal/.well-known/jwks.json"
audience = "switchboard"
issuer = "https://auth.internal"

[auth.validators.api_keys]
type = "static_keys"
keys = ["${SWITCHBOARD_API_KEY_1}", "${SWITCHBOARD_API_KEY_2}"]

# ─── Upstream Providers + Key Pools ───

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_format = "anthropic"
models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"]
health_check_interval = "30s"
timeout = "300s"
max_concurrent = 100

[providers.anthropic.key_pool]
selector = "weighted_random"

[[providers.anthropic.key_pool.keys]]
id = "anthropic-1"
api_key = "${ANTHROPIC_API_KEY_1}"
weight = 1.0

[[providers.anthropic.key_pool.keys]]
id = "anthropic-2"
api_key = "${ANTHROPIC_API_KEY_2}"
weight = 1.0

[providers.openai]
base_url = "https://api.openai.com"
api_format = "openai"
models = ["gpt-4o", "gpt-4o-mini", "o1", "o3"]

[providers.openai.key_pool]
selector = "weighted_random"

[[providers.openai.key_pool.keys]]
id = "openai-1"
api_key = "${OPENAI_API_KEY_1}"
weight = 1.0

[providers.bedrock]
api_format = "bedrock"
region = "us-east-1"
cross_region_inference = true
models = [
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "amazon.nova-pro-v1:0",
]

[providers.bedrock.key_pool]
selector = "round_robin"

[[providers.bedrock.key_pool.keys]]
id = "bedrock-east"
type = "aws_sts"
role_arn = "arn:aws:iam::123456789:role/switchboard"
region = "us-east-1"
refresh_interval = "45m"

[[providers.bedrock.key_pool.keys]]
id = "bedrock-west"
type = "aws_sts"
role_arn = "arn:aws:iam::123456789:role/switchboard"
region = "us-west-2"
refresh_interval = "45m"

[providers.ollama]
base_url = "http://gpu-box.internal:11434"
api_format = "openai"
models = ["llama3.3:70b", "codestral:latest"]

[providers.ollama.key_pool]
selector = "round_robin"
# Ollama typically has no auth, but the pool still tracks health

# ─── Model Selection ───

[model_selection]
mode = "dynamic"
header = "X-Switchboard-Model"
fallback = "claude-sonnet-4-20250514"

# ─── Semantic Routing ───

[routing.semantic]
enabled = true
classifier = "heuristic"

[[routing.semantic.rules]]
task_type = "code_generation"
complexity = "complex"
preferred_models = ["claude-opus-4-20250514", "o3"]

[[routing.semantic.rules]]
task_type = "code_generation"
complexity = "simple"
preferred_models = ["claude-haiku-4-5-20251001", "gpt-4o-mini"]

[routing.semantic.default]
preferred_models = ["claude-sonnet-4-20250514"]

# ─── Guardrails ───

[guardrails]
enabled = true
fail_mode = "open"
timeout = "500ms"
streaming_mode = "async_audit"

[[guardrails.engines]]
type = "builtin_secret_detection"
phase = "pre_request"
action = "block"

[[guardrails.engines]]
type = "grpc"
phase = "pre_request"
endpoint = "guardrails-service.internal:50051"
tls = true
timeout = "200ms"

# ─── Observability ───

[observability]
enabled = true
exporter = "otlp"
otlp_endpoint = "http://localhost:4317"
service_name = "switchboard"
environment = "production"
sample_rate = 1.0
capture_prompts = false
batch_flush_interval = "5s"

# ─── Rate Limits ───

[rate_limit]
enabled = true
default_rpm = 60
default_tpm = 100000

[rate_limit.overrides.power-users]
rpm = 120
tpm = 500000
```

---

## 14. Test Harness

### 14.1 Testing Philosophy

Every layer of the system is testable in isolation via traits and dependency injection. Integration tests use `wiremock` to simulate upstream LLM providers and external guardrail services.

### 14.2 Test Categories

**Unit Tests** (`cargo test --lib`):
- Auth provider credential lifecycle (obtain, cache, refresh, expire)
- Key pool selection algorithms (weighted random, round robin, least loaded, sticky)
- Key health tracking (error rate calculation, rate limit detection, status transitions)
- Identity resolution from various sources
- Model selection policy evaluation
- Semantic classifier (heuristic mode — task detection, complexity estimation)
- Guardrail engine evaluation (regex, keyword, secret detection)
- Request/response transformation correctness (OpenAI, Anthropic, Bedrock formats)
- Bedrock SigV4 request signing
- Configuration parsing and validation
- Cost calculation accuracy
- switchboard-local config parsing and model override logic

**Integration Tests** (`cargo test --test '*'`):
- Full proxy round-trip (non-streaming): client → switchboard → wiremock upstream → client
- Full proxy round-trip (streaming): SSE event-by-event verification
- switchboard-local → switchboard-server round-trip (auth + proxy)
- Auth injection: verify correct headers reach upstream
- Auth refresh: simulate 401 → refresh → retry flow with key pool fallback
- Key pool rotation: simulate rate limit on key-1 → automatic switch to key-2
- Key pool health: verify degraded keys are deprioritized
- Model override: verify model name transformation in upstream request
- Semantic routing: verify prompt classification → model selection
- Identity propagation: verify user identity appears in trace spans
- Multi-provider failover: primary down → fallback provider used
- Bedrock end-to-end: verify SigV4 signing, Converse API transform, streaming
- Rate limiting: verify 429 responses when limits exceeded
- Graceful shutdown: in-flight streams complete, new requests rejected

**Guardrail Tests** (`cargo test --test guardrails`):
- Built-in regex engine: PII detection, secret detection, keyword blocking
- Built-in token limit engine: verify enforcement per user/team
- gRPC callout: mock gRPC evaluator, verify request/response contract
- HTTP callout: mock HTTP evaluator, verify request/response contract
- Pipeline ordering: verify pre-request runs before LLM, post-response runs after
- Fail-open mode: verify requests pass when guardrail engine is unavailable
- Fail-closed mode: verify requests block when guardrail engine is unavailable
- Streaming guardrails: verify buffered, chunked, and async_audit modes
- Guardrail + observability: verify guardrail spans are emitted correctly

**Admin API Tests** (`cargo test --test admin_api`):
- Key pool CRUD: add/remove/update keys via API, verify pool behavior changes
- Model selection updates: change policy via API, verify routing changes
- Guardrail config updates: enable/disable engines, verify evaluation changes
- Rate limit updates: modify limits via API, verify enforcement
- Usage stats: verify accurate aggregation by user/model/team
- Config reload: verify hot-reload applies new config
- Auth: verify admin endpoints require admin auth, reject proxy auth

**Observability Tests** (`cargo test --test observability`):
- Verify span structure matches DD LLM Obs expectations
- Verify GenAI semantic convention attributes are set correctly
- Verify token counts are extracted from streaming responses
- Verify `time_to_first_token` measurement accuracy
- Verify spans are properly closed on client disconnect
- Verify cost calculation for various models and providers
- Verify guardrail spans are included in the trace
- Verify semantic routing decision is recorded in span attributes

**Snapshot Tests** (using `insta`):
- Serialized request transformations (OpenAI → Bedrock, OpenAI → Anthropic, etc.)
- Serialized response transformations
- Admin API response payloads
- Configuration serialization round-trips
- Guardrail verdict serialization

**Property Tests** (using `proptest`):
- Request transformation preserves all non-modified fields
- SSE stream parsing handles arbitrary chunk boundaries
- Auth credential refresh is thread-safe under concurrent access
- Key pool selection never returns disabled keys
- Semantic classifier produces valid classification for any input

### 14.3 Test Infrastructure

```rust
// Shared test helpers
mod test_helpers {
    /// Start a wiremock server simulating an OpenAI-compatible upstream
    pub async fn mock_openai_upstream() -> MockServer { ... }

    /// Start a wiremock server simulating an Anthropic upstream
    pub async fn mock_anthropic_upstream() -> MockServer { ... }

    /// Start a wiremock server simulating Bedrock Converse API
    pub async fn mock_bedrock_upstream() -> MockServer { ... }

    /// Start a mock gRPC guardrail evaluator
    pub async fn mock_grpc_guardrail() -> MockGuardrailServer { ... }

    /// Start a mock HTTP guardrail evaluator
    pub async fn mock_http_guardrail() -> MockServer { ... }

    /// Start a switchboard-server instance pointed at mock upstreams
    pub async fn start_server(config: ServerConfig) -> TestServer { ... }

    /// Start a switchboard-local instance pointed at a test server
    pub async fn start_local(config: LocalConfig) -> TestLocalProxy { ... }

    /// Generate a valid SSE stream for chat completions
    pub fn sse_chat_stream(chunks: &[&str]) -> String { ... }

    /// Assert that a span was emitted with expected attributes
    pub fn assert_span(spans: &[SpanData], name: &str, attrs: &[(&str, &str)]) { ... }

    /// Create a key pool with N mock keys at various health states
    pub fn mock_key_pool(healthy: usize, degraded: usize, disabled: usize) -> KeyPool { ... }
}
```

---

## 15. Technology Stack

| Component | Crate | Version | Purpose |
|-----------|-------|---------|---------|
| HTTP Framework | `axum` | 0.8.x | Request handling, routing, SSE responses |
| Async Runtime | `tokio` | 1.x | Async I/O, timers, channels |
| HTTP Client | `reqwest` | 0.12.x | Upstream provider calls, streaming |
| Tower Middleware | `tower` + `tower-http` | 0.5.x / 0.6.x | Auth, rate limiting, tracing, timeouts |
| SSE Client | `reqwest-eventsource` | latest | Parse upstream SSE streams |
| SSE Types | `eventsource-stream` | latest | Low-level SSE event parsing |
| gRPC | `tonic` | 0.12.x | Guardrail gRPC client + proto codegen |
| Protobuf | `prost` | 0.13.x | Protobuf serialization for gRPC |
| AWS SigV4 | `aws-sigv4` + `aws-credential-types` | latest | Bedrock request signing |
| AWS STS | `aws-sdk-sts` | latest | Temporary credential generation for Bedrock |
| Tracing | `tracing` + `tracing-subscriber` | latest | Structured logging and span creation |
| OTel Bridge | `tracing-opentelemetry` | latest | Bridge tracing spans to OTel |
| OTel SDK | `opentelemetry` + `opentelemetry-sdk` | 0.31.x | OpenTelemetry core |
| OTel Export | `opentelemetry-otlp` | latest | OTLP gRPC/HTTP export |
| DD OTel | `datadog-opentelemetry` | 0.2.x | Datadog-specific OTel optimizations |
| JWT | `jsonwebtoken` | latest | JWT decode/validate |
| Serialization | `serde` + `serde_json` | latest | JSON serialization |
| Config | `config` | 0.15.x | Layered configuration |
| CLI | `clap` | 4.x | Command-line arguments |
| Env | `dotenvy` | latest | `.env` file loading |
| Static Assets | `rust-embed` | latest | Embed admin UI in binary |
| Testing | `wiremock` | latest | HTTP mock server |
| gRPC Testing | `tonic` mock | — | In-process gRPC mock server |
| Snapshots | `insta` | latest | Snapshot testing |
| Property | `proptest` | latest | Property-based testing |

---

## 16. Performance Targets

| Metric | Target |
|--------|--------|
| Proxy overhead (P50, no guardrails) | < 500us |
| Proxy overhead (P99, no guardrails) | < 1ms |
| Proxy overhead (P99, with built-in guardrails) | < 2ms |
| Proxy overhead (P99, with external gRPC guardrail) | < guardrail_timeout + 1ms |
| Memory (idle) | < 20MB |
| Memory (1k concurrent streams) | < 100MB |
| Max concurrent connections | 10,000+ |
| Time to first byte (added) | < 1ms (no guardrails), < guardrail latency (with) |
| Server binary size (release) | < 35MB |
| Local binary size (release) | < 10MB |
| Startup time | < 500ms |

---

## 17. Future Considerations

These are explicitly **out of scope** for v1 but noted for future reference:

- **Prompt caching**: Cache common system prompts or few-shot examples
- **Multi-region**: Deploy Switchboard server instances close to providers with global routing
- **Cost budgets**: Per-user or per-team spending limits with alerts and automatic cutoff
- **Provider-specific optimizations**: Anthropic prompt caching headers, OpenAI batch API
- **Response caching**: Cache identical requests for deterministic workloads (temperature=0)
- **A/B model testing**: Route a percentage of traffic to a new model and compare quality metrics
- **Embedding-based semantic classifier**: Upgrade from heuristic to vector similarity classification
