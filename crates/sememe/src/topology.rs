use serde::{Deserialize, Serialize};

use crate::path::ModulePath;

/// The model's module topology as exposed by a `Backend`. M1 ships the
/// type; only the stub produces it today. The bridge produces it in M2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleTree {
    pub root: String,
    pub children: Vec<ModulePath>,
}
