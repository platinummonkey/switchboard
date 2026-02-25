// Switchboard protocol header names injected by switchboard-local.

pub const HEADER_USER: &str = "x-switchboard-user";
pub const HEADER_TEAM: &str = "x-switchboard-team";
pub const HEADER_MODEL: &str = "x-switchboard-model";
pub const HEADER_REQUEST_ID: &str = "x-switchboard-request-id";
pub const HEADER_TOOL: &str = "x-switchboard-tool";

pub const PROTOCOL_VERSION: &str = "1";

/// All Switchboard-specific header names, for iteration and stripping.
pub const ALL_HEADERS: &[&str] = &[
    HEADER_USER,
    HEADER_TEAM,
    HEADER_MODEL,
    HEADER_REQUEST_ID,
    HEADER_TOOL,
];

/// Returns true if the given header name (case-insensitive) is a Switchboard protocol header.
pub fn is_switchboard_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ALL_HEADERS.iter().any(|h| *h == lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_names_are_lowercase() {
        for h in ALL_HEADERS {
            assert_eq!(*h, h.to_ascii_lowercase(), "header {h} must be lowercase");
        }
    }

    #[test]
    fn test_is_switchboard_header_exact() {
        assert!(is_switchboard_header(HEADER_USER));
        assert!(is_switchboard_header(HEADER_TEAM));
        assert!(is_switchboard_header(HEADER_MODEL));
        assert!(is_switchboard_header(HEADER_REQUEST_ID));
        assert!(is_switchboard_header(HEADER_TOOL));
    }

    #[test]
    fn test_is_switchboard_header_case_insensitive() {
        assert!(is_switchboard_header("X-Switchboard-User"));
        assert!(is_switchboard_header("X-SWITCHBOARD-MODEL"));
    }

    #[test]
    fn test_is_not_switchboard_header() {
        assert!(!is_switchboard_header("authorization"));
        assert!(!is_switchboard_header("content-type"));
        assert!(!is_switchboard_header("x-forwarded-for"));
    }

    #[test]
    fn test_all_headers_count() {
        // Ensures we don't silently drop a header from ALL_HEADERS.
        assert_eq!(ALL_HEADERS.len(), 5);
    }

    #[test]
    fn test_protocol_version_nonempty() {
        assert!(!PROTOCOL_VERSION.is_empty());
    }
}
