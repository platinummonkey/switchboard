-- Initial schema for switchboard-server persistence layer.
--
-- Key metadata only — no credential values are stored here.
-- Static keys stay in TOML config.  Dynamic keys (aws_sts, vault) store
-- their source parameters (role_arn / vault_path / region) in source_config.

CREATE TABLE IF NOT EXISTS key_pool_entries (
    provider_id     TEXT             NOT NULL,
    id              TEXT             NOT NULL,
    key_type        TEXT             NOT NULL,  -- "static" | "aws_sts" | "vault"
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    status          TEXT             NOT NULL DEFAULT 'healthy',
    source_config   JSONB            NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider_id, id)
);

CREATE TABLE IF NOT EXISTS rate_limit_overrides (
    id          TEXT        PRIMARY KEY,
    rpm         INTEGER     NOT NULL,
    tpm         INTEGER     NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS config_overrides (
    section     TEXT    PRIMARY KEY
        CHECK (section IN ('model_selection', 'guardrails', 'routing')),
    config_json JSONB   NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
