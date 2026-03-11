//! Diesel schema definitions for the three persistence tables.
//!
//! Maintained by hand — matches the SQL in `migrations/0001_initial.sql`.

diesel::table! {
    key_pool_entries (provider_id, id) {
        provider_id -> Text,
        id -> Text,
        key_type -> Text,
        weight -> Float8,
        status -> Text,
        source_config -> Jsonb,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    rate_limit_overrides (id) {
        id -> Text,
        rpm -> Integer,
        tpm -> Integer,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    config_overrides (section) {
        section -> Text,
        config_json -> Jsonb,
        updated_at -> Timestamptz,
    }
}
