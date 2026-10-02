use std::fmt;

use serde::{Deserialize, Serialize};

/// A dotted path into a model's module tree, e.g.
/// `"encoder.layer.3.attention"`. Newtype so paths stay distinct from
/// arbitrary strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModulePath(String);

impl ModulePath {
    /// Build a path from an existing dotted string. Does not validate the
    /// segments — the bridge is the source of truth in M2+; for M1 the stub
    /// passes hand-built paths.
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Segments of the path, in order. Empty segments are preserved.
    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.0.split('.').filter(|s| !s.is_empty())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModulePath {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ModulePath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl AsRef<str> for ModulePath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
