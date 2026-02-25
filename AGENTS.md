# Switchboard — Agent Instructions

## Project Overview

Switchboard is a two-component LLM proxy system written in **Rust**:

- **`switchboard-server`** — Centralized gateway that routes requests to upstream LLM providers (Anthropic, OpenAI, Google, Amazon Bedrock, Ollama) with dynamic key pool management, semantic routing, pluggable prompt guardrails, user identity tracking, and native Datadog LLM Observability.
- **`switchboard-local`** — Lightweight client-side proxy (~5MB) that developers run locally. Handles auth to the server, presents OpenAI/Anthropic-compatible endpoints on localhost, and injects model preferences + user identity.

**This is NOT a multi-tenant platform.** It is a single-deployment, team-oriented system with an admin-only control plane (API + UI).

## Architecture

See [docs/DESIGN.md](docs/DESIGN.md) for the full design document including competitive analysis, component architecture, and integration patterns.

### Key Design Decisions

- **Language**: Rust (latest stable edition 2024)
- **Workspace**: Cargo workspace with three crates (`switchboard-server`, `switchboard-local`, `switchboard-common`)
- **HTTP framework**: `axum` 0.8.x with `tower` middleware
- **Async runtime**: `tokio` 1.x (multi-threaded)
- **HTTP client**: `reqwest` 0.12.x with streaming
- **gRPC**: `tonic` 0.12.x (guardrail callouts)
- **AWS**: `aws-sigv4` + `aws-sdk-sts` (Bedrock integration)
- **Observability**: OpenTelemetry GenAI Semantic Conventions → Datadog Agent OTLP → DD LLM Observability
- **Config format**: TOML with env var overrides via `config` crate
- **Admin UI**: Embedded static SPA via `rust-embed`, optional at compile time
- **Testing**: `wiremock` for HTTP mocks, `tonic` for gRPC mocks, `insta` for snapshots, `proptest` for property tests

### Project Structure

```
switchboard/
├── crates/
│   ├── switchboard-server/        # Central gateway server
│   │   └── src/
│   │       ├── main.rs
│   │       ├── config/            # Configuration types and loading
│   │       ├── auth/              # Client→server auth (JWT, mTLS, API key validators)
│   │       ├── key_pool/          # Dynamic upstream key management + rotation
│   │       ├── identity/          # User identity extraction
│   │       ├── routing/           # Model/provider selection + semantic routing
│   │       ├── guardrails/        # Pluggable guardrail pipeline (builtin + gRPC/HTTP callout)
│   │       ├── proxy/             # Core proxy logic, SSE streaming
│   │       ├── providers/         # Per-provider modules (openai, anthropic, bedrock, vertex, ollama)
│   │       ├── observability/     # DD LLM Obs, OTel, metrics
│   │       ├── admin/             # Admin REST API + embedded UI
│   │       └── middleware/        # Tower layers
│   ├── switchboard-local/         # Client-side proxy for developer machines
│   │   └── src/
│   │       ├── main.rs
│   │       ├── config.rs          # ~/.switchboard/config.toml
│   │       ├── auth.rs            # Auth to switchboard-server (JWT refresh, mTLS)
│   │       ├── server.rs          # Local HTTP server on localhost
│   │       ├── proxy.rs           # Forward to switchboard-server
│   │       └── model_prefs.rs     # Local model preference injection
│   └── switchboard-common/        # Shared types
│       └── src/
│           ├── lib.rs
│           ├── types.rs           # Shared request/response types
│           ├── protocol.rs        # Switchboard protocol headers
│           └── models.rs          # Model name registry and aliases
├── tests/
│   ├── integration/               # End-to-end tests (proxy, auth, streaming, guardrails, admin API, local proxy)
│   ├── unit/                      # Per-module unit tests
│   └── fixtures/                  # Request/response/guardrail fixtures
├── proto/
│   └── guardrails/v1/
│       └── evaluator.proto        # Guardrail gRPC service definition
├── admin-ui/                      # Admin UI (embedded static SPA)
├── config/                        # Default and example configs
├── Cargo.toml                     # Workspace root
└── docs/
    └── DESIGN.md
```

## Code Conventions

### General

- Use `thiserror` for error types, `anyhow` only in `main.rs` / CLI
- All public types derive `Debug`. Domain types also derive `Clone`, `Serialize`, `Deserialize` as appropriate
- Use `tracing` for all logging (never `println!` or `eprintln!` in library code)
- Prefer `impl Trait` in return position over boxed trait objects where possible
- All async code uses `tokio`. No `async-std` or `smol`

### Naming

- Modules use `snake_case`
- Types use `PascalCase`
- Constants use `SCREAMING_SNAKE_CASE`
- Trait methods that return futures are `async fn` (via `#[async_trait]` or RPITIT on nightly)

### Error Handling

- Library code returns `Result<T, SwitchboardError>` (custom enum)
- Proxy handlers return `Result<Response, StatusCode>` or use `IntoResponse`
- Never panic in request handling paths. Use `?` propagation
- Log errors at the middleware level, not in individual providers

### Testing

- Every public function or trait method has at least one unit test
- Integration tests cover every API endpoint (streaming and non-streaming)
- Use `wiremock::MockServer` for upstream LLM provider simulation
- Use `tonic` mock servers for gRPC guardrail evaluator testing
- Use `insta` snapshots for request/response transformation assertions
- Fixtures live in `tests/fixtures/` as `.json` files
- Test names follow `test_<module>_<scenario>_<expected>` pattern
- All tests must pass with `cargo test` — no manual setup required

### Dependencies

- Pin major versions in `Cargo.toml` (e.g., `axum = "0.8"`)
- Run `cargo clippy -- -D warnings` before committing
- Run `cargo fmt` before committing
- No `unsafe` code without a safety comment and approval

## Key Traits

### AuthProvider

```rust
#[async_trait]
pub trait AuthProvider: Send + Sync + 'static {
    fn name(&self) -> &str;
    async fn get_credentials(&self) -> Result<UpstreamCredentials, AuthError>;
    async fn refresh(&self) -> Result<UpstreamCredentials, AuthError>;
    fn is_valid(&self) -> bool;
}
```

Implement to add new client→server auth mechanisms. Register in the `AuthRegistry`.

### KeySelector

```rust
pub trait KeySelector: Send + Sync {
    fn select(&self, pool: &[PooledKey], request: &RequestContext) -> Option<&PooledKey>;
}
```

Implement to add new key pool selection strategies. Built-in: `WeightedRandom`, `RoundRobin`, `LeastLoaded`, `Sticky`.

### GuardrailEngine

```rust
#[async_trait]
pub trait GuardrailEngine: Send + Sync {
    fn name(&self) -> &str;
    async fn evaluate_request(&self, input: &GuardrailInput) -> Result<GuardrailVerdict>;
    async fn evaluate_response(&self, input: &GuardrailInput) -> Result<GuardrailVerdict>;
}
```

Implement to add new guardrail evaluation logic. Built-in: `RegexEngine`, `KeywordEngine`, `TokenLimitEngine`, `SecretDetectionEngine`. External: `GrpcCalloutEngine`, `HttpCalloutEngine`.

### SemanticClassifier

```rust
#[async_trait]
pub trait SemanticClassifier: Send + Sync {
    async fn classify(&self, input: &ClassificationInput) -> ClassificationResult;
}
```

Implement to add new prompt classification strategies. Built-in: `HeuristicClassifier`.

### IdentityResolver

Chain of resolvers that extract user identity from request context. Add new resolvers by implementing the `IdentityResolver` trait.

## Observability

### Datadog LLM Observability

We use OpenTelemetry GenAI Semantic Conventions (v1.37+) exported via OTLP to the Datadog Agent. Key attributes on every LLM span:

- `gen_ai.operation.name` — "chat", "completion", "embedding"
- `gen_ai.provider.name` — "anthropic", "openai", "bedrock", etc.
- `gen_ai.request.model` / `gen_ai.response.model`
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
- Custom: `switchboard.user.id`, `switchboard.user.team`, `switchboard.tool`
- Custom: `switchboard.key_pool.key_id`, `switchboard.model.selection_reason`
- Custom: `switchboard.guardrail.pre_action`, `switchboard.guardrail.post_action`

### Tracing

Use `tracing` spans throughout. The `TraceLayer` from `tower-http` instruments HTTP requests. Add custom spans for auth resolution, key pool selection, guardrail evaluation, routing decisions, and stream lifecycle events.

## Tool Integration

With `switchboard-local` running, all tools point to `localhost:8877`:

| Tool | Config |
|------|--------|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8877/api` |
| Cursor | OpenAI base URL: `http://localhost:8877/v1` |
| OpenCode | OpenAI base URL: `http://localhost:8877/v1` |
| Continue | OpenAI base URL: `http://localhost:8877/v1` |
| Aider | `--openai-api-base http://localhost:8877/v1` |

## Admin API

The admin control plane is API-first. Every operation is available via REST at `/admin/api/v1/`. The admin UI is a thin SPA over these APIs. Key endpoints:

- Key pool management: `/admin/api/v1/providers/{id}/keys`
- Model selection: `/admin/api/v1/model-selection`
- Guardrails: `/admin/api/v1/guardrails`
- Semantic routing: `/admin/api/v1/routing/semantic`
- Usage stats: `/admin/api/v1/usage`
- Rate limits: `/admin/api/v1/rate-limits`
- Config reload: `POST /admin/api/v1/config/reload`

## Performance Targets

- Proxy overhead (no guardrails): < 1ms P99
- Proxy overhead (built-in guardrails): < 2ms P99
- Memory (idle): < 20MB
- Max concurrent streams: 10,000+
- Server binary size: < 35MB
- Local binary size: < 10MB

## CI/CD

- `cargo fmt --check` — formatting
- `cargo clippy -- -D warnings` — lints
- `cargo test` — all tests
- `cargo build --release` — release binaries (server + local)
