//! Ratatui-based TUI that renders the sememe live hub against any
//! `Backend`. Three panes: module tree on the left, selected module's
//! `TensorView` in the middle, recent `HubEvent`s on the right. Vim keys
//! for navigation. M4 milestone — first user-visible iteration.

pub mod state;
pub mod view;

pub mod app;
pub mod ndjson;

pub use state::{TuiEvent, TuiState};
pub use view::{Pane, RenderModel};

use sememe::harness::Harness;
use sememe::{Result, backend::Backend};

/// Run the TUI against the given harness. Blocks until the user quits.
/// On exit the TUI unsubscribes from the hub and restores the terminal.
pub fn run<B: Backend>(mut harness: Harness<B>) -> Result<()> {
    app::run(&mut harness)
}
