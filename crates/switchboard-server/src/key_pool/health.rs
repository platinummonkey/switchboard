//! Per-key health metadata tracked at runtime.

use std::time::Instant;

// ── Key status ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyStatus {
    /// No elevated error rate; eligible for full-weight selection.
    Healthy,
    /// Elevated errors; eligible for selection at reduced weight.
    Degraded,
    /// Hit provider rate limits recently; deprioritised for a backoff window.
    RateLimited,
    /// Administratively or automatically disabled; never selected.
    Disabled,
}

impl KeyStatus {
    /// Returns `true` if the key may be selected to serve requests.
    pub fn is_eligible(&self) -> bool {
        matches!(
            self,
            KeyStatus::Healthy | KeyStatus::Degraded | KeyStatus::RateLimited
        )
    }

    /// Returns `true` if the key is available without penalty.
    pub fn is_healthy(&self) -> bool {
        matches!(self, KeyStatus::Healthy)
    }
}

impl std::fmt::Display for KeyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyStatus::Healthy => write!(f, "healthy"),
            KeyStatus::Degraded => write!(f, "degraded"),
            KeyStatus::RateLimited => write!(f, "rate_limited"),
            KeyStatus::Disabled => write!(f, "disabled"),
        }
    }
}

// ── Key health snapshot ───────────────────────────────────────────────────────

/// Runtime health metrics for a single pooled key.
/// Updated by the proxy layer after each request.
#[derive(Debug, Clone)]
pub struct KeyHealth {
    pub total_requests: u64,
    pub errors_last_5m: u64,
    pub rate_limit_hits_last_5m: u64,
    pub avg_latency_ms: f64,
    pub last_used: Option<Instant>,
    pub last_error: Option<(Instant, String)>,
    pub status: KeyStatus,
}

impl Default for KeyHealth {
    fn default() -> Self {
        Self {
            total_requests: 0,
            errors_last_5m: 0,
            rate_limit_hits_last_5m: 0,
            avg_latency_ms: 0.0,
            last_used: None,
            last_error: None,
            status: KeyStatus::Healthy,
        }
    }
}

impl KeyHealth {
    /// Effective selection weight multiplier based on current status.
    /// Healthy = 1.0, Degraded = 0.3, RateLimited = 0.0, Disabled = 0.0.
    pub fn weight_multiplier(&self) -> f64 {
        match self.status {
            KeyStatus::Healthy => 1.0,
            KeyStatus::Degraded => 0.3,
            KeyStatus::RateLimited | KeyStatus::Disabled => 0.0,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_eligibility() {
        assert!(KeyStatus::Healthy.is_eligible());
        assert!(KeyStatus::Degraded.is_eligible());
        assert!(KeyStatus::RateLimited.is_eligible());
        assert!(!KeyStatus::Disabled.is_eligible());
    }

    #[test]
    fn test_status_is_healthy() {
        assert!(KeyStatus::Healthy.is_healthy());
        assert!(!KeyStatus::Degraded.is_healthy());
        assert!(!KeyStatus::RateLimited.is_healthy());
        assert!(!KeyStatus::Disabled.is_healthy());
    }

    #[test]
    fn test_weight_multiplier() {
        let mut h = KeyHealth::default();
        assert!((h.weight_multiplier() - 1.0).abs() < f64::EPSILON);

        h.status = KeyStatus::Degraded;
        assert!((h.weight_multiplier() - 0.3).abs() < f64::EPSILON);

        h.status = KeyStatus::RateLimited;
        assert!((h.weight_multiplier() - 0.0).abs() < f64::EPSILON);

        h.status = KeyStatus::Disabled;
        assert!((h.weight_multiplier() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_health_is_healthy() {
        let h = KeyHealth::default();
        assert_eq!(h.status, KeyStatus::Healthy);
        assert_eq!(h.total_requests, 0);
        assert!(h.last_used.is_none());
        assert!(h.last_error.is_none());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(KeyStatus::Healthy.to_string(), "healthy");
        assert_eq!(KeyStatus::Degraded.to_string(), "degraded");
        assert_eq!(KeyStatus::RateLimited.to_string(), "rate_limited");
        assert_eq!(KeyStatus::Disabled.to_string(), "disabled");
    }
}
