use crate::backend::Backend;
use crate::edit::EditOp;
use crate::error::Result;
use crate::input::ModelInput;
use crate::path::ModulePath;
use crate::telemetry::Telemetry;
use crate::view::TensorView;

/// Conductor between a `Backend` and the live `Telemetry` hub. Every
/// `forward` and `edit` call goes through here, so subscribers see the
/// same events the backend returns.
pub struct Harness<B: Backend> {
    backend: B,
    telemetry: Telemetry,
}

impl<B: Backend> Harness<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            telemetry: Telemetry::new(),
        }
    }

    pub fn new_with_telemetry(backend: B, telemetry: Telemetry) -> Self {
        Self { backend, telemetry }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    pub fn telemetry_mut(&mut self) -> &mut Telemetry {
        &mut self.telemetry
    }

    /// Ask the backend for the module tree and record it on the hub.
    pub fn topology(&mut self) -> Result<crate::topology::ModuleTree> {
        let tree = self.backend.named_modules()?;
        self.telemetry.record_topology(&tree)?;
        Ok(tree)
    }

    /// Run a forward pass, record every returned `TensorView` on the
    /// hub, and return them. Bumps the hub's forward counter so
    /// `forwards_since_last_obs` is accurate.
    pub fn forward(&mut self, input: &ModelInput) -> Result<Vec<TensorView>> {
        self.telemetry.begin_forward();
        let views = self.backend.run_forward(input)?;
        for view in views {
            self.telemetry.record_observation(view);
        }
        Ok(self.collect_views())
    }

    /// Apply an edit through the backend, then record it on the hub.
    pub fn edit(&mut self, path: &ModulePath, op: &EditOp) -> Result<()> {
        self.backend.edit(path, op)?;
        self.telemetry.record_edit(path, op.clone());
        Ok(())
    }

    fn collect_views(&self) -> Vec<TensorView> {
        let mut out = Vec::new();
        for node in self.telemetry.iter() {
            if let Some(view) = &node.latest {
                out.push(view.clone());
            }
        }
        out
    }
}
