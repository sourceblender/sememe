//! Ratatui runtime. Owns the terminal, subscribes to the harness's
//! telemetry, and drives `view::render` on every event.

use std::io::{Stdout, stdout};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::event::{self, Event as CtEvent, KeyCode, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};

use sememe::backend::Backend;
use sememe::harness::Harness;
use sememe::{Error as CoreError, Result as CoreResult};

use crate::state::TuiState;
use crate::view::{MiddlePane, RenderModel, render};

pub fn run<B: Backend>(harness: &mut Harness<B>) -> CoreResult<()> {
    let mut terminal = setup_terminal().map_err(|e| CoreError::Backend(e.to_string()))?;

    let state = Arc::new(Mutex::new(TuiState::new()));

    // Subscriber pushes events into the right pane buffer.
    let state_for_sub = Arc::clone(&state);
    harness.telemetry().subscribe(move |event| {
        if let Ok(mut s) = state_for_sub.lock() {
            s.push_event(event.clone());
        }
    });

    let result = event_loop(&mut terminal, harness, state);
    teardown_terminal(&mut terminal).map_err(|e| CoreError::Backend(e.to_string()))?;
    result
}

fn event_loop<B: Backend>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    harness: &mut Harness<B>,
    state: Arc<Mutex<TuiState>>,
) -> CoreResult<()> {
    loop {
        {
            let nodes: Vec<&_> = harness.telemetry().iter().collect();
            let snapshot = {
                let s = state.lock().expect("tui state lock");
                if s.quit {
                    return Ok(());
                }
                render(&nodes, &s, harness.telemetry().current_forward())
            };
            draw(terminal, &snapshot)?;
        }

        if event::poll(Duration::from_millis(200))? {
            if let CtEvent::Key(KeyEvent { code, kind, .. }) = event::read()? {
                if kind != KeyEventKind::Press {
                    continue;
                }
                let nodes_len = harness.telemetry().iter().count();
                let mut s = state.lock().expect("tui state lock");
                handle_key(code, &mut s, nodes_len);
            }
        }
    }
}

fn handle_key(code: KeyCode, state: &mut TuiState, node_count: usize) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => state.quit = true,
        KeyCode::Char('j') | KeyCode::Down => state.move_selection(1, node_count),
        KeyCode::Char('k') | KeyCode::Up => state.move_selection(-1, node_count),
        KeyCode::Char('g') => state.move_selection(-(node_count as i64), node_count),
        KeyCode::Char('G') => state.move_selection(node_count as i64, node_count),
        _ => {}
    }
}

fn draw(terminal: &mut Terminal<CrosstermBackend<Stdout>>, model: &RenderModel) -> CoreResult<()> {
    terminal
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(30),
                    Constraint::Percentage(45),
                    Constraint::Percentage(25),
                ])
                .split(f.area());

            let tree_items: Vec<ListItem> = model
                .tree
                .iter()
                .map(|row| {
                    let indent = "  ".repeat(row.depth);
                    let style = if row.selected {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(Line::from(Span::styled(
                        format!("{indent}{}", row.label),
                        style,
                    )))
                })
                .collect();
            let tree = List::new(tree_items).block(
                Block::default()
                    .title("Tree")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            f.render_widget(tree, chunks[0]);

            let middle_widget = match &model.middle {
                MiddlePane::Empty => Paragraph::new("(no module selected)").block(
                    Block::default()
                        .title("Selected")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray)),
                ),
                MiddlePane::Node(node) => {
                    let mut lines: Vec<Line> = Vec::new();
                    lines.push(Line::from(format!("path: {}", node.path)));
                    if let Some(view) = &node.latest {
                        lines.push(Line::from(format!(
                            "shape: {:?}  dtype: {}",
                            view.shape, view.dtype
                        )));
                        lines.push(Line::from(format!(
                            "min: {:.6}  max: {:.6}  mean: {:.6}",
                            view.min, view.max, view.mean
                        )));
                        let samples = view
                            .samples
                            .iter()
                            .map(|s| format!("{s:+.4}"))
                            .collect::<Vec<_>>()
                            .join("  ");
                        lines.push(Line::from(format!("samples: {samples}")));
                    } else {
                        lines.push(Line::from("(no observation yet)"));
                    }
                    lines.push(Line::from(format!(
                        "history: {}   edits: {}   mean_delta: {}   forwards_since_last: {}",
                        node.history_len,
                        node.edits_len,
                        match node.mean_delta {
                            Some(d) => format!("{d:+.6}"),
                            None => "—".to_string(),
                        },
                        node.forwards_since_last_obs,
                    )));
                    Paragraph::new(lines).block(
                        Block::default()
                            .title("Selected")
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::DarkGray)),
                    )
                }
            };
            f.render_widget(middle_widget, chunks[1]);

            let event_items: Vec<ListItem> = model
                .events
                .iter()
                .map(|s| ListItem::new(Line::from(s.as_str())))
                .collect();
            let events = List::new(event_items).block(
                Block::default()
                    .title("Events")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            f.render_widget(events, chunks[2]);
        })
        .map_err(|e| CoreError::Backend(format!("ratatui draw: {e}")))?;
    Ok(())
}

fn setup_terminal() -> std::io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(out);
    Terminal::new(backend)
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> std::io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
