//! Serde helpers for human-readable duration strings ("500ms", "30s", "45m", "2h").
#![allow(dead_code)]

use std::time::Duration;

use serde::{Deserialize, Deserializer, Serializer};

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
    let s = String::deserialize(d)?;
    parse(&s).map_err(serde::de::Error::custom)
}

pub fn serialize<S: Serializer>(dur: &Duration, s: S) -> Result<S::Ok, S::Error> {
    let ms = dur.as_millis();
    let text = if ms % 3_600_000 == 0 {
        format!("{}h", ms / 3_600_000)
    } else if ms % 60_000 == 0 {
        format!("{}m", ms / 60_000)
    } else if ms % 1_000 == 0 {
        format!("{}s", ms / 1_000)
    } else {
        format!("{}ms", ms)
    };
    s.serialize_str(&text)
}

pub fn deserialize_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
    let opt = Option::<String>::deserialize(d)?;
    match opt {
        None => Ok(None),
        Some(s) => parse(&s).map(Some).map_err(serde::de::Error::custom),
    }
}

pub fn parse(s: &str) -> Result<Duration, String> {
    if let Some(v) = s.strip_suffix("ms") {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| format!("invalid duration: {s}"))?;
        Ok(Duration::from_millis(n))
    } else if let Some(v) = s.strip_suffix('s') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| format!("invalid duration: {s}"))?;
        Ok(Duration::from_secs(n))
    } else if let Some(v) = s.strip_suffix('m') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| format!("invalid duration: {s}"))?;
        Ok(Duration::from_secs(n * 60))
    } else if let Some(v) = s.strip_suffix('h') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| format!("invalid duration: {s}"))?;
        Ok(Duration::from_secs(n * 3600))
    } else {
        Err(format!(
            "invalid duration '{s}': expected suffix ms, s, m, or h"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ms() {
        assert_eq!(parse("500ms").unwrap(), Duration::from_millis(500));
    }

    #[test]
    fn test_parse_seconds() {
        assert_eq!(parse("30s").unwrap(), Duration::from_secs(30));
    }

    #[test]
    fn test_parse_minutes() {
        assert_eq!(parse("45m").unwrap(), Duration::from_secs(45 * 60));
    }

    #[test]
    fn test_parse_hours() {
        assert_eq!(parse("2h").unwrap(), Duration::from_secs(7200));
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse("abc").is_err());
        assert!(parse("123").is_err());
        assert!(parse("").is_err());
    }
}
