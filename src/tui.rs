use crate::api::{ApiConfig, Provider};
use crate::distill::{DistillConfig, OutputFormat, ProgressEvent, Sample};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap},
    Frame,
};
use std::time::Duration;
use tokio::sync::mpsc;

const ACCENT: Color = Color::Rgb(108, 143, 255);
const GREEN: Color = Color::Rgb(52, 199, 89);
const RED: Color = Color::Rgb(255, 59, 48);
const YELLOW: Color = Color::Rgb(255, 149, 0);
const DIM: Color = Color::DarkGray;

#[derive(Clone, PartialEq)]
pub enum Screen {
    Setup,
    Running,
    Done,
}

pub struct App {
    pub screen: Screen,
    pub api_config: ApiConfig,
    pub distill_config: DistillConfig,
    pub tab: usize,
    pub field_idx: usize,  // campo selezionato nel tab corrente
    pub input_buffer: String,
    pub editing: bool,

    // Running state
    pub progress: f64,
    pub phase: String,
    pub total_target: usize,
    pub completed: usize,
    pub errors: usize,
    pub samples: Vec<Sample>,
    pub log: Vec<String>,
    pub log_scroll: usize,
    pub running: bool,

    // Done state
    pub final_path: String,
    pub final_count: usize,

    // Quit confirmation
    pub confirm_quit: bool,

    // Checkpoint
    pub checkpoint_interval: usize,
    pub last_checkpoint: usize,

    pub rx: Option<mpsc::UnboundedReceiver<ProgressEvent>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            screen: Screen::Setup,
            api_config: ApiConfig::default(),
            distill_config: DistillConfig::default(),
            tab: 0,
            field_idx: 0,
            input_buffer: String::new(),
            editing: false,
            progress: 0.0,
            phase: String::new(),
            total_target: 0,
            completed: 0,
            errors: 0,
            samples: Vec::new(),
            log: Vec::new(),
            log_scroll: 0,
            running: false,
            final_path: String::new(),
            final_count: 0,
            confirm_quit: false,
            checkpoint_interval: 100,
            last_checkpoint: 0,
            rx: None,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode) -> bool {
        // Quit confirmation dialog
        if self.confirm_quit {
            match code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    // If running, save checkpoint before quitting
                    if self.screen == Screen::Running && !self.samples.is_empty() {
                        self.save_checkpoint();
                    }
                    return true;
                }
                _ => {
                    self.confirm_quit = false;
                    return false;
                }
            }
        }

        // Editing mode
        if self.editing {
            match code {
                KeyCode::Esc => { self.editing = false; }
                KeyCode::Enter => {
                    self.apply_edit();
                    self.editing = false;
                }
                KeyCode::Backspace => { self.input_buffer.pop(); }
                KeyCode::Char(c) => { self.input_buffer.push(c); }
                _ => {}
            }
            return false;
        }

        match self.screen {
            Screen::Setup => match code {
                KeyCode::Char('q') | KeyCode::Esc => { self.confirm_quit = true; }
                KeyCode::Tab => { self.tab = (self.tab + 1) % 4; self.field_idx = 0; }
                KeyCode::BackTab => { self.tab = if self.tab == 0 { 3 } else { self.tab - 1 }; self.field_idx = 0; }
                KeyCode::Char('1') => { self.tab = 0; self.field_idx = 0; }
                KeyCode::Char('2') => { self.tab = 1; self.field_idx = 0; }
                KeyCode::Char('3') => { self.tab = 2; self.field_idx = 0; }
                KeyCode::Char('4') => { self.tab = 3; self.field_idx = 0; }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.field_idx > 0 { self.field_idx -= 1; }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let max = self.max_fields();
                    if self.field_idx < max.saturating_sub(1) { self.field_idx += 1; }
                }
                KeyCode::Enter => self.handle_enter(),
                _ => {}
            },
            Screen::Running => match code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.confirm_quit = true;
                }
                KeyCode::Up => { if self.log_scroll > 0 { self.log_scroll -= 1; } }
                KeyCode::Down => { self.log_scroll += 1; }
                _ => {}
            },
            Screen::Done => match code {
                KeyCode::Char('q') | KeyCode::Esc => { self.confirm_quit = true; }
                KeyCode::Enter | KeyCode::Char('n') => {
                    self.screen = Screen::Setup;
                    self.samples.clear();
                    self.log.clear();
                    self.completed = 0;
                    self.errors = 0;
                }
                _ => {}
            },
        }
        false
    }

    fn save_checkpoint(&self) {
        let path = format!(
            "{}.checkpoint",
            self.distill_config.output_path.to_string_lossy()
        );
        match crate::distill::save_dataset(
            &self.samples,
            &self.distill_config.output_format,
            std::path::Path::new(&path),
        ) {
            Ok(n) => {
                eprintln!("Checkpoint saved: {} samples → {}", n, path);
            }
            Err(e) => {
                eprintln!("Checkpoint save failed: {}", e);
            }
        }
    }

    fn max_fields(&self) -> usize {
        match self.tab {
            0 => 5,  // Provider, Model, API Key, Max Tokens, Temperature
            1 => 4,  // Topics, Samples, Language, System Prompt
            2 => 3,  // Format, Output Path, Parallel
            3 => 1,  // Run button
            _ => 0,
        }
    }

    fn handle_enter(&mut self) {
        match self.tab {
            0 => match self.field_idx {
                0 => { // Cicla provider
                    let providers = Provider::all();
                    let idx = providers.iter().position(|p| p.name() == self.api_config.provider.name()).unwrap_or(0);
                    self.api_config.provider = providers[(idx + 1) % providers.len()].clone();
                    self.api_config.model = self.api_config.provider.default_model().into();
                }
                1 => { self.start_edit(&self.api_config.model.clone()); }
                2 => { self.start_edit(&self.api_config.api_key.clone()); }
                3 => { self.start_edit(&self.api_config.max_tokens.to_string()); }
                4 => { self.start_edit(&format!("{:.1}", self.api_config.temperature)); }
                _ => {}
            },
            1 => match self.field_idx {
                0 => { self.start_edit(&self.distill_config.topics.join(", ")); }
                1 => { self.start_edit(&self.distill_config.n_samples.to_string()); }
                2 => { self.start_edit(&self.distill_config.lang.clone()); }
                3 => { self.start_edit(&self.distill_config.system_prompt.clone()); }
                _ => {}
            },
            2 => match self.field_idx {
                0 => { // Cicla formato
                    let fmts = OutputFormat::all();
                    let idx = fmts.iter().position(|f| f.name() == self.distill_config.output_format.name()).unwrap_or(0);
                    self.distill_config.output_format = fmts[(idx + 1) % fmts.len()].clone();
                }
                1 => { let p = self.distill_config.output_path.to_string_lossy().to_string(); self.start_edit(&p); }
                2 => { self.start_edit(&self.distill_config.parallel.to_string()); }
                _ => {}
            },
            3 => {
                if !self.api_config.api_key.is_empty() {
                    self.running = true;
                }
            }
            _ => {}
        }
    }

    fn start_edit(&mut self, current: &str) {
        self.input_buffer = current.to_string();
        self.editing = true;
    }

    fn apply_edit(&mut self) {
        let val = self.input_buffer.clone();
        match self.tab {
            0 => match self.field_idx {
                1 => self.api_config.model = val,
                2 => self.api_config.api_key = val,
                3 => { if let Ok(n) = val.parse() { self.api_config.max_tokens = n; } }
                4 => { if let Ok(f) = val.parse() { self.api_config.temperature = f; } }
                _ => {}
            },
            1 => match self.field_idx {
                0 => {
                    self.distill_config.topics = val.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                1 => { if let Ok(n) = val.parse() { self.distill_config.n_samples = n; } }
                2 => self.distill_config.lang = val,
                3 => self.distill_config.system_prompt = val,
                _ => {}
            },
            2 => match self.field_idx {
                1 => self.distill_config.output_path = val.into(),
                2 => { if let Ok(n) = val.parse() { self.distill_config.parallel = n; } }
                _ => {}
            },
            _ => {}
        }
    }

    pub fn process_events(&mut self) {
        if let Some(rx) = &mut self.rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    ProgressEvent::Phase(p) => {
                        self.phase = p.clone();
                        self.log.push(format!("── {p}"));
                    }
                    ProgressEvent::QuestionGenerated(idx, q) => {
                        let display: String = q.chars().take(80).collect();
                        self.log.push(format!("  Q{idx}: {display}"));
                    }
                    ProgressEvent::AnswerGenerated(idx, sample) => {
                        self.completed += 1;
                        self.progress = self.completed as f64 / self.total_target.max(1) as f64;
                        let display: String = sample.answer.chars().take(60).collect();
                        self.log.push(format!(
                            "  ✓ A{idx}: {display} ({} tok)",
                            sample.tokens_est
                        ));
                        self.samples.push(sample);

                        // Auto-checkpoint every N samples
                        if self.checkpoint_interval > 0
                            && self.completed > 0
                            && self.completed % self.checkpoint_interval == 0
                            && self.completed != self.last_checkpoint
                        {
                            self.last_checkpoint = self.completed;
                            self.save_checkpoint();
                            self.log.push(format!(
                                "── 💾 Checkpoint saved ({} samples)",
                                self.completed
                            ));
                        }
                    }
                    ProgressEvent::Error(idx, msg) => {
                        self.errors += 1;
                        self.log.push(format!("  ✗ E{idx}: {msg}"));
                    }
                    ProgressEvent::Done(total) => {
                        self.running = false;
                        self.final_count = total;
                        self.final_path = self.distill_config.output_path.to_string_lossy().into();
                        self.screen = Screen::Done;
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
//  RENDERING
// ══════════════════════════════════════════════════════════════════════

pub fn draw(f: &mut Frame, app: &App) {
    let size = f.area();

    // Background nero
    f.render_widget(Block::default().style(Style::default().bg(Color::Rgb(0,0,0))), size);

    // Header
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(1)])
        .split(size);

    // Title bar
    let title = Line::from(vec![
        Span::styled(" ⚗ ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("alembic", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::styled(" ", Style::default().fg(DIM)),
    ]);
    let title_block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM)).style(Style::default().bg(Color::Rgb(0,0,0)));
    f.render_widget(Paragraph::new(title).block(title_block), chunks[0]);

    // Status bar
    let status = match app.screen {
        Screen::Setup => "Tab: switch │ Enter: select │ q: quit".to_string(),
        Screen::Running => format!(
            "↑/↓: scroll │ q: stop │ checkpoint every {} samples",
            app.checkpoint_interval
        ),
        Screen::Done => "Enter: new run │ q: quit".to_string(),
    };
    let status = status.as_str();
    f.render_widget(
        Paragraph::new(Span::styled(format!(" {status}"), Style::default().fg(DIM))),
        chunks[2],
    );

    // Main content
    match app.screen {
        Screen::Setup => draw_setup(f, app, chunks[1]),
        Screen::Running => draw_running(f, app, chunks[1]),
        Screen::Done => draw_done(f, app, chunks[1]),
    }

    // Quit confirmation overlay
    if app.confirm_quit {
        let msg = if app.screen == Screen::Running && !app.samples.is_empty() {
            format!(
                " Quit? {} samples will be saved to checkpoint. (y/n) ",
                app.samples.len()
            )
        } else {
            " Quit? (y/n) ".to_string()
        };

        let popup = centered_rect(50, 3, size);
        let confirm = Paragraph::new(Line::from(vec![
            Span::styled(&msg, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]))
        .alignment(ratatui::layout::Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(RED))
                .style(Style::default().bg(Color::Rgb(30, 0, 0))),
        );
        f.render_widget(Block::default().style(Style::default().bg(Color::Rgb(0, 0, 0))), popup);
        f.render_widget(confirm, popup);
    }
}

fn draw_setup(f: &mut Frame, app: &App, area: Rect) {
    let tabs = Tabs::new(vec![
            Line::raw("1 Provider"),
            Line::raw("2 Topics"),
            Line::raw("3 Format"),
            Line::raw("4 Run"),
        ])
        .select(app.tab)
        .style(Style::default().fg(DIM))
        .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .divider(Span::raw("│"));

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(area);

    f.render_widget(tabs, inner[0]);

    match app.tab {
        0 => draw_provider_tab(f, app, inner[1]),
        1 => draw_topics_tab(f, app, inner[1]),
        2 => draw_format_tab(f, app, inner[1]),
        3 => draw_run_tab(f, app, inner[1]),
        _ => {}
    }
}

fn draw_provider_tab(f: &mut Frame, app: &App, area: Rect) {
    let fields = vec![
        ("Provider", app.api_config.provider.name().to_string(), "Enter: cycle"),
        ("Model", app.api_config.model.clone(), "Enter: edit"),
        ("API Key", mask_key(&app.api_config.api_key), "Enter: edit"),
        ("Max Tokens", app.api_config.max_tokens.to_string(), "Enter: edit"),
        ("Temperature", format!("{:.1}", app.api_config.temperature), "Enter: edit"),
    ];
    let items: Vec<ListItem> = fields.iter().enumerate().map(|(i, (label, value, hint))| {
        let sel = i == app.field_idx;
        let marker = if sel { "▸ " } else { "  " };
        let style = if sel { Style::default().fg(ACCENT).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) };
        let hint_style = if sel { Style::default().fg(DIM) } else { Style::default().fg(Color::Rgb(0,0,0)) };
        ListItem::new(Line::from(vec![
            Span::styled(marker, Style::default().fg(ACCENT)),
            Span::styled(format!("{label:16}"), Style::default().fg(DIM)),
            Span::styled(value.clone(), style),
            Span::styled(format!("  {hint}"), hint_style),
        ]))
    }).collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" API Configuration ↑↓ Navigate  Enter: Select ").border_style(Style::default().fg(DIM)));
    f.render_widget(list, area);

    if app.editing {
        draw_edit_overlay(f, app, area);
    }
}

fn draw_topics_tab(f: &mut Frame, app: &App, area: Rect) {
    let fields = vec![
        ("Topics", app.distill_config.topics.join(", "), "Enter: edit"),
        ("Samples", app.distill_config.n_samples.to_string(), "Enter: edit"),
        ("Language", app.distill_config.lang.clone(), "Enter: edit"),
        ("System Prompt", truncate_safe(&app.distill_config.system_prompt, 50), "Enter: edit"),
    ];
    let items: Vec<ListItem> = fields.iter().enumerate().map(|(i, (label, value, hint))| {
        let sel = i == app.field_idx;
        let marker = if sel { "▸ " } else { "  " };
        let style = if sel { Style::default().fg(ACCENT).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) };
        let hint_style = if sel { Style::default().fg(DIM) } else { Style::default().fg(Color::Rgb(0,0,0)) };
        ListItem::new(Line::from(vec![
            Span::styled(marker, Style::default().fg(ACCENT)),
            Span::styled(format!("{label:16}"), Style::default().fg(DIM)),
            Span::styled(value.clone(), style),
            Span::styled(format!("  {hint}"), hint_style),
        ]))
    }).collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Content Settings ↑↓ Navigate  Enter: Select ").border_style(Style::default().fg(DIM)));
    f.render_widget(list, area);

    if app.editing {
        draw_edit_overlay(f, app, area);
    }
}

fn draw_format_tab(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    let fields = vec![
        ("Format", app.distill_config.output_format.name().to_string(), "Enter: cycle"),
        ("Output File", app.distill_config.output_path.to_string_lossy().to_string(), "Enter: edit"),
        ("Parallel", app.distill_config.parallel.to_string(), "Enter: edit"),
    ];
    let items: Vec<ListItem> = fields.iter().enumerate().map(|(i, (label, value, hint))| {
        let sel = i == app.field_idx;
        let marker = if sel { "▸ " } else { "  " };
        let style = if sel { Style::default().fg(ACCENT).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) };
        let hint_style = if sel { Style::default().fg(DIM) } else { Style::default().fg(Color::Rgb(0,0,0)) };
        ListItem::new(Line::from(vec![
            Span::styled(marker, Style::default().fg(ACCENT)),
            Span::styled(format!("{label:16}"), Style::default().fg(DIM)),
            Span::styled(value.clone(), style),
            Span::styled(format!("  {hint}"), hint_style),
        ]))
    }).collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Output ").border_style(Style::default().fg(DIM)));
    f.render_widget(list, cols[0]);

    // Preview
    let preview = app.distill_config.output_format.format_pair(
        "What is the capital of France?",
        "The capital of France is Paris.",
    );
    let prev_widget = Paragraph::new(preview)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(" Preview ").border_style(Style::default().fg(DIM)));
    f.render_widget(prev_widget, cols[1]);

    if app.editing {
        draw_edit_overlay(f, app, area);
    }
}

fn draw_run_tab(f: &mut Frame, app: &App, area: Rect) {
    let summary = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Provider:  ", Style::default().fg(DIM)),
            Span::styled(app.api_config.provider.name(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Model:     ", Style::default().fg(DIM)),
            Span::styled(&app.api_config.model, Style::default().fg(ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("  Samples:   ", Style::default().fg(DIM)),
            Span::styled(app.distill_config.n_samples.to_string(), Style::default().fg(GREEN)),
        ]),
        Line::from(vec![
            Span::styled("  Topics:    ", Style::default().fg(DIM)),
            Span::raw(app.distill_config.topics.join(", ")),
        ]),
        Line::from(vec![
            Span::styled("  Format:    ", Style::default().fg(DIM)),
            Span::raw(app.distill_config.output_format.name()),
        ]),
        Line::from(vec![
            Span::styled("  Output:    ", Style::default().fg(DIM)),
            Span::raw(app.distill_config.output_path.to_string_lossy()),
        ]),
        Line::from(vec![
            Span::styled("  API Key:   ", Style::default().fg(DIM)),
            Span::styled(
                if app.api_config.api_key.is_empty() { "⚠ NOT SET" } else { "✓ set" },
                Style::default().fg(if app.api_config.api_key.is_empty() { RED } else { GREEN }),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press Enter to start distillation",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
    ];
    let p = Paragraph::new(summary)
        .block(Block::default().borders(Borders::ALL).title(" Summary — Ready to Run ").border_style(Style::default().fg(DIM)));
    f.render_widget(p, area);
}

fn draw_running(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // Progress
    let pct = (app.progress * 100.0) as u16;
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!(" {} ", app.phase)).border_style(Style::default().fg(DIM)))
        .gauge_style(Style::default().fg(ACCENT))
        .percent(pct.min(100))
        .label(format!("{}/{} completed, {} errors", app.completed, app.total_target, app.errors));
    f.render_widget(gauge, chunks[0]);

    // Log
    let log_items: Vec<ListItem> = app.log.iter()
        .skip(app.log_scroll)
        .map(|l| {
            let style = if l.starts_with("  ✓") {
                Style::default().fg(GREEN)
            } else if l.starts_with("  ✗") {
                Style::default().fg(RED)
            } else if l.starts_with("──") {
                Style::default().fg(YELLOW).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(DIM)
            };
            ListItem::new(Span::styled(l, style))
        })
        .collect();
    let log_list = List::new(log_items)
        .block(Block::default().borders(Borders::ALL).title(" Log ").border_style(Style::default().fg(DIM)));
    f.render_widget(log_list, chunks[1]);

    // Stats
    let stats = Line::from(vec![
        Span::styled("  Tokens: ", Style::default().fg(DIM)),
        Span::styled(
            format!("~{}", app.samples.iter().map(|s| s.tokens_est).sum::<usize>()),
            Style::default().fg(ACCENT),
        ),
        Span::styled("  │  Samples: ", Style::default().fg(DIM)),
        Span::styled(app.samples.len().to_string(), Style::default().fg(GREEN)),
    ]);
    f.render_widget(
        Paragraph::new(stats).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM))),
        chunks[2],
    );
}

fn draw_done(f: &mut Frame, app: &App, area: Rect) {
    let total_tokens: usize = app.samples.iter().map(|s| s.tokens_est).sum();
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled("  ✓ Distillation Complete!", Style::default().fg(GREEN).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Samples:  ", Style::default().fg(DIM)),
            Span::styled(app.final_count.to_string(), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Tokens:   ", Style::default().fg(DIM)),
            Span::styled(format!("~{total_tokens}"), Style::default().fg(ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("  Errors:   ", Style::default().fg(DIM)),
            Span::styled(app.errors.to_string(), Style::default().fg(if app.errors > 0 { RED } else { GREEN })),
        ]),
        Line::from(vec![
            Span::styled("  Saved to: ", Style::default().fg(DIM)),
            Span::styled(&app.final_path, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Enter: new run │ q: quit", Style::default().fg(DIM))),
    ];
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Results ").border_style(Style::default().fg(GREEN)));
    f.render_widget(p, area);
}

fn draw_edit_overlay(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(60, 5, area);
    let input = Paragraph::new(Line::from(vec![
        Span::styled("> ", Style::default().fg(ACCENT)),
        Span::raw(&app.input_buffer),
        Span::styled("█", Style::default().fg(Color::White)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Edit (Enter to confirm, Esc to cancel) ")
            .border_style(Style::default().fg(ACCENT)),
    );
    f.render_widget(Block::default().style(Style::default().bg(Color::Rgb(0,0,0))), popup);
    f.render_widget(input, popup);
}

// Helpers

fn list_item(label: &str, value: &str, highlight: bool) -> ListItem<'static> {
    let style = if highlight {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(Color::White)
    };
    ListItem::new(Line::from(vec![
        Span::styled(format!("  {label:16}"), Style::default().fg(DIM)),
        Span::styled(value.to_string(), style),
    ]))
}

fn mask_key(key: &str) -> String {
    if key.is_empty() {
        "not set".into()
    } else if key.len() <= 8 {
        "*".repeat(key.len())
    } else {
        format!("{}...{}", &key[..4], &key[key.len()-4..])
    }
}

fn truncate_safe(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect::<String>() + "…"
    }
}

fn centered_rect(percent_x: u16, height: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length((area.height.saturating_sub(height)) / 2),
            Constraint::Length(height),
            Constraint::Min(0),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
