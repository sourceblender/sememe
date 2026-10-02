//! `sememe` command-line interface.
//!
//! M4 ships two commands:
//! - no args: print the version (placeholder until M5).
//! - `tui`: load the demo model and run the live TUI against it.

use std::process::ExitCode;

use sememe::Harness;
use sememe_bridge::PyBackend;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("--version") | Some("-v") => {
            println!("sememe {}", sememe::version());
            ExitCode::SUCCESS
        }
        Some("tui") => run_tui(),
        Some("help") | Some("-h") | Some("--help") => {
            println!(
                "sememe — debug harness for transformer models\n\n\
                 USAGE:\n  sememe                 print the version\n  \
                 sememe tui             launch the live TUI against the demo model\n  \
                 sememe --help          show this help\n"
            );
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}\nrun `sememe --help` for usage.");
            ExitCode::from(2)
        }
    }
}

fn run_tui() -> ExitCode {
    let backend = match PyBackend::load(sememe_bridge::demo_model_path()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "failed to load model at {}: {e}\n\
                 Make sure the demo model is on disk and the venv is set up.\n\
                 See README.md for setup instructions.",
                sememe_bridge::demo_model_path()
            );
            return ExitCode::from(1);
        }
    };

    let mut harness = Harness::new(backend);
    if let Err(e) = harness.topology() {
        eprintln!("topology failed: {e}");
        return ExitCode::from(1);
    }

    if let Err(e) = sememe_tui::run(harness) {
        eprintln!("tui exited with error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
