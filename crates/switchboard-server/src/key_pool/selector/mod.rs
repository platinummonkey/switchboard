//! Concrete [`KeySelector`] implementations.
//!
//! | Selector | Strategy |
//! |---|---|
//! | [`WeightedRandomSelector`] | Weighted random over `effective_weight()` |
//! | [`RoundRobinSelector`] | Ordered cycling through eligible keys |
//! | [`LeastLoadedSelector`] | Pick key with the lowest `total_requests` |
//! | [`StickySelector`] | Per-user affinity wrapping another selector |

pub mod least_loaded;
pub mod round_robin;
pub mod sticky;
pub mod weighted_random;

pub use least_loaded::LeastLoadedSelector;
pub use round_robin::RoundRobinSelector;
pub use sticky::StickySelector;
pub use weighted_random::WeightedRandomSelector;
