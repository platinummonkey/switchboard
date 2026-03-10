# Switchboard

A two-component LLM proxy system that gives your team visibility and control over every AI request — without changing how developers work.

---

## What is it?

Switchboard sits between your developers' tools and the LLM providers (Anthropic, OpenAI, Bedrock, Vertex, Ollama). It solves the problems that show up once more than a handful of people start using AI tools:

- **Credential sprawl** — API keys checked into dotfiles, shared over Slack, rotated manually across dozens of machines
- **No visibility** — you don't know who's using what model, how much it costs, or what prompts are going out
- **No guardrails** — nothing stops sensitive data from being sent to external models
- **Provider lock-in** — switching providers means updating every tool on every machine
- **Rate limit chaos** — one runaway process hammers the API and blocks everyone else

The system has two parts:

```
Developer's machine                     Your infrastructure
┌─────────────────┐                    ┌──────────────────────────┐
│                 │  auth + headers    │                          │
│  Your AI tools  │───────────────────▶│  switchboard-server      │
│  (Claude Code,  │                    │  ├── auth                │
│   Cursor, etc.) │                    │  ├── guardrails          │──▶ Anthropic
│        │        │                    │  ├── routing             │──▶ OpenAI
│        ▼        │                    │  ├── key pool            │──▶ Bedrock
│  switchboard-   │                    │  ├── rate limiting       │──▶ Vertex
│  local:8877     │                    │  ├── observability       │──▶ Ollama
│                 │                    │  └── admin API + UI      │
└─────────────────┘                    └──────────────────────────┘
```

**`switchboard-local`** is a ~5MB binary developers run on their machines. It handles auth to the server and presents OpenAI/Anthropic-compatible endpoints on `localhost:8877`. Developers point their tools at localhost — no API keys on their machines.

**`switchboard-server`** is the centralized gateway your team runs. It holds the real credentials, enforces guardrails, tracks usage, and routes requests to the right upstream provider.

---

## Getting started

### Prerequisites

- Rust 1.85+
- `protoc` (Protocol Buffers compiler) for gRPC support
- PostgreSQL (optional — only needed for persistent config storage)

### Build from source

```bash
git clone https://github.com/DataDog/switchboard
cd switchboard

# Install protoc (macOS)
brew install protobuf

# Build both binaries
cargo build --release

# Binaries land here:
# target/release/switchboard-server
# target/release/switchboard-local
```

### Run with Podman/Docker Compose

The quickest way to spin up the full stack locally:

```bash
# Copy the example environment file and fill in your API keys
cp deploy/server.toml config/switchboard-server.toml
# Edit config/switchboard-server.toml with your provider API keys

# Start everything (PostgreSQL + server + local proxy)
podman compose up
# or: docker compose up
```

This starts:
| Service | Port | Purpose |
|---|---|---|
| PostgreSQL | 5432 | Persistent config storage |
| switchboard-server (proxy) | 8080 | LLM proxy gateway |
| switchboard-server (admin) | 9090 | Admin API + UI |
| switchboard-local | 8877 | Developer-facing local proxy |

---

## Developer setup (switchboard-local)

Once the server is running, developers set up their local proxy:

```bash
# Interactive setup wizard
switchboard-local init
# Prompts for: server URL, auth method, your identity (name/team)
# Writes config to ~/.switchboard/config.toml

# Start the local proxy
switchboard-local start

# Check it's working
switchboard-local status
```

Then point your tools at `localhost:8877`:

| Tool | Config |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8877/api` |
| Cursor | OpenAI base URL: `http://localhost:8877/v1` |
| Continue | OpenAI base URL: `http://localhost:8877/v1` |
| OpenCode | OpenAI base URL: `http://localhost:8877/v1` |
| Aider | `--openai-api-base http://localhost:8877/v1` |

No API keys on developer machines. The local proxy authenticates to the server on their behalf.

---

## Server configuration

Copy `config/switchboard-server.example.toml` and customize it. The minimal config to get started:

```toml
[server]
listen = "0.0.0.0:8080"

[admin]
enabled = true
listen = "127.0.0.1:9090"

[[auth.validators]]
type = "static_token"
token = "your-shared-api-key"

[[providers]]
name = "anthropic"
type = "anthropic"
api_key = "sk-ant-..."

[[providers]]
name = "openai"
type = "openai"
api_key = "sk-..."
```

See `config/switchboard-server.example.toml` for the full reference including model selection, semantic routing, guardrails, rate limiting, observability, and PostgreSQL persistence.

### Auth methods

The server supports several ways for `switchboard-local` to authenticate:

| Method | Good for |
|---|---|
| Static API key | Simple shared-token setups |
| JWT | SSO-integrated environments, user identity from claims |
| mTLS | High-security, certificate-based identity |
| OAuth2 client credentials | Service account flows |

### Providers

Out of the box: **Anthropic**, **OpenAI**, **AWS Bedrock** (SigV4 + cross-region inference), **Google Vertex AI**, **Ollama** (local models).

Providers are defined in config. You can have multiple instances of the same provider type (e.g., different Bedrock regions, different OpenAI organizations).

### Key pool management

Instead of a single API key per provider, you can configure a pool:

```toml
[[providers.anthropic.key_pool]]
key = "sk-ant-team-key-1"
weight = 70

[[providers.anthropic.key_pool]]
key = "sk-ant-team-key-2"
weight = 30
```

Selection strategies: `weighted_random`, `round_robin`, `least_loaded`, `sticky`. Keys rotate automatically on errors.

---

## Guardrails

Guardrails run on every request (and optionally every response) before traffic reaches the upstream provider. They can **block**, **warn**, or **pass** requests.

Built-in engines:
- **regex** — match patterns in prompt text
- **keyword** — exact keyword matching
- **token_limit** — cap max tokens per request
- **secret_detection** — catch API keys, passwords, and other secrets in prompts

External callout engines:
- **gRPC** — call your own guardrail service (proto definition in `proto/guardrails/v1/`)
- **HTTP** — call any HTTP endpoint

```toml
[guardrails]
enabled = true
fail_mode = "open"   # "open" = pass on error, "closed" = block on error

[[guardrails.engines]]
type = "secret_detection"
action = "block"

[[guardrails.engines]]
type = "grpc"
address = "your-guardrail-service:50051"
action = "block"
```

---

## Routing and model selection

### Static model mappings

Map generic model names to provider-specific ones, or redirect all requests for one model to another:

```toml
[model_selection]
mode = "dynamic"

[model_selection.mappings]
"claude-3-5-sonnet" = { provider = "anthropic", model = "claude-3-5-sonnet-20241022" }
"gpt-4o" = { provider = "openai", model = "gpt-4o-2024-11-20" }
```

### Semantic routing

Route requests to different models based on what the prompt is asking for:

```toml
[[routing.semantic.rules]]
task_type = "code"
complexity = "high"
target_model = "claude-3-5-sonnet"

[[routing.semantic.rules]]
task_type = "summarization"
complexity = "low"
target_model = "gpt-4o-mini"
```

### Per-tool model overrides

Set different defaults for different tools. Developers can still override in their local config.

---

## Rate limiting

```toml
[rate_limit]
requests_per_minute = 500
tokens_per_minute = 500000

[[rate_limit.overrides]]
group = "heavy-users-team"
requests_per_minute = 1000
tokens_per_minute = 2000000
```

---

## Observability

Switchboard emits OpenTelemetry spans using the GenAI semantic conventions, exported via OTLP to your Datadog Agent → Datadog LLM Observability.

Every LLM request gets a span with:
- `gen_ai.operation.name`, `gen_ai.provider.name`
- `gen_ai.request.model` / `gen_ai.response.model`
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
- `switchboard.user.id`, `switchboard.user.team`, `switchboard.tool`
- `switchboard.guardrail.pre_action`, `switchboard.guardrail.post_action`

```toml
[observability]
enabled = true
otlp_endpoint = "http://datadog-agent:4317"
sample_rate = 1.0
```

---

## Admin API and UI

The admin control plane runs on a separate listener (`127.0.0.1:9090` by default, localhost-only). Every operation is available via REST:

| Endpoint | Purpose |
|---|---|
| `/admin/api/v1/providers/{id}/keys` | Manage key pools |
| `/admin/api/v1/model-selection` | Configure model routing |
| `/admin/api/v1/guardrails` | Configure guardrail engines |
| `/admin/api/v1/routing/semantic` | Semantic routing rules |
| `/admin/api/v1/usage` | Usage statistics |
| `/admin/api/v1/rate-limits` | Rate limit config and overrides |
| `/admin/api/v1/config/reload` | Hot-reload config from disk |

The embedded admin UI is available at `http://localhost:9090/admin/` (when built with the `admin-ui` feature).

---

## PostgreSQL persistence

By default Switchboard runs entirely in-memory — config lives in TOML files. Enable PostgreSQL to persist admin mutations across restarts:

```toml
[database]
enabled = true
url = "postgres://switchboard:password@localhost:5432/switchboard"
```

On startup with a database configured, Switchboard runs migrations automatically and loads any stored overrides on top of the file-based config. If the database is unreachable and `enabled = true`, the server refuses to start (fail-fast rather than silently losing state).

---

## Performance

| Metric | Target |
|---|---|
| Proxy overhead (no guardrails) | < 1ms P99 |
| Proxy overhead (built-in guardrails) | < 2ms P99 |
| Memory at idle | < 20MB |
| Max concurrent streams | 10,000+ |
| Server binary size | < 35MB |
| Local binary size | < 10MB |

---

## Development

```bash
# Run all tests
cargo test

# Lint (must be clean before committing)
cargo clippy -- -D warnings

# Format (must be clean before committing)
cargo fmt

# Build release binaries
cargo build --release
```

The test suite uses:
- `wiremock` for simulating upstream LLM providers
- `tonic` mock servers for gRPC guardrail testing
- `insta` for snapshot testing request/response transformations
- `proptest` for property-based tests
- Real TCP listener-based E2E tests (no mocking the server itself)

See `docs/DESIGN.md` for the full architecture, design decisions, and competitive analysis.

---

## Project structure

```
switchboard/
├── crates/
│   ├── switchboard-server/     # Central gateway
│   ├── switchboard-local/      # Developer machine proxy
│   └── switchboard-common/     # Shared types
├── proto/guardrails/v1/        # gRPC service definition
├── admin-ui/                   # Embedded admin SPA
├── config/                     # Default + example configs
├── deploy/                     # Deployment entrypoints
├── tests/                      # Integration tests + fixtures
├── docs/DESIGN.md              # Full design document
├── Containerfile               # Multi-stage container build
└── compose.yaml                # Podman/Docker Compose stack
```

---

## License

Apache 2.0 — see LICENSE.

Built by the Datadog Platform Team.
