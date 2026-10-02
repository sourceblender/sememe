use serde::{Deserialize, Serialize};

/// Input fed to a forward pass. Minimal on purpose — text-encoder inputs
/// are token ids and (optionally) token type ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInput {
    pub ids: Vec<i64>,
    pub type_ids: Option<Vec<i64>>,
}

impl ModelInput {
    pub fn from_ids(ids: impl Into<Vec<i64>>) -> Self {
        Self {
            ids: ids.into(),
            type_ids: None,
        }
    }
}
