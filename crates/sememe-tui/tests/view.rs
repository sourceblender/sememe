use sememe::HubEvent;
use sememe::{EditOp, ModulePath, ModuleTree, Telemetry, TensorDType, TensorView};
use sememe_tui::TuiState;
use sememe_tui::view::MiddlePane;
use sememe_tui::view::render;

fn view(path: &str, mean: f64) -> TensorView {
    TensorView::from_slice(
        ModulePath::new(path),
        vec![3],
        TensorDType::Float32,
        &[mean as f32 - 0.1, mean as f32, mean as f32 + 0.1],
    )
}

#[test]
fn empty_telemetry_renders_root_in_middle() {
    let t = Telemetry::new();
    let nodes: Vec<&_> = t.iter().collect();
    assert_eq!(nodes.len(), 1, "telemetry always has a root node");
    let s = TuiState::new();
    let m = render(&nodes, &s, 0);
    // Root has no `latest` (no observations yet) but is still a Node,
    // not Empty. Verify we render it.
    match m.middle {
        MiddlePane::Node(n) => {
            assert!(n.latest.is_none());
            assert_eq!(n.path, "");
        }
        _ => panic!("expected Node (root)"),
    }
}

#[test]
fn render_includes_all_nodes_preorder() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a.b", 1.0));
    t.record_observation(view("a.c", 2.0));

    let nodes: Vec<&_> = t.iter().collect();
    let s = TuiState::new();
    let m = render(&nodes, &s, 1);

    assert_eq!(m.tree.len(), 4);
    assert_eq!(m.tree[0].label, "");
    assert_eq!(m.tree[1].label, "a");
    assert_eq!(m.tree[2].label, "b");
    assert_eq!(m.tree[3].label, "c");
}

#[test]
fn selection_marks_one_tree_row() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a", 1.0));
    t.record_observation(view("b", 2.0));

    let nodes: Vec<&_> = t.iter().collect();
    let mut state = TuiState::new();
    state.move_selection(2, nodes.len());
    let m = render(&nodes, &state, 1);

    let count_selected = m.tree.iter().filter(|r| r.selected).count();
    assert_eq!(count_selected, 1);
}

#[test]
fn middle_pane_reflects_selected_node() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a", 1.5));

    let nodes: Vec<&_> = t.iter().collect();
    // Selected = the "a" node (index 1; 0 is the empty root).
    let mut s = TuiState::new();
    s.move_selection(1, nodes.len());
    let m = render(&nodes, &s, 1);

    match m.middle {
        MiddlePane::Node(n) => {
            assert_eq!(n.path, "a");
            assert!(n.latest.is_some());
            assert_eq!(n.history_len, 1);
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn events_buffer_tracks_hub_activity() {
    let mut s = TuiState::new();
    s.push_event(HubEvent::Topology {
        tree: ModuleTree {
            root: "model".into(),
            children: vec![ModulePath::new("a")],
        },
    });
    s.push_event(HubEvent::Observation {
        path: ModulePath::new("a"),
        view: view("a", 0.5),
    });
    s.push_event(HubEvent::Edit {
        path: ModulePath::new("a"),
        op: EditOp::Zero,
        before: None,
        after: Box::new(None),
    });

    assert_eq!(s.events.len(), 3);
}
