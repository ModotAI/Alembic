mod api;
mod distill;
mod tui;

use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;
use std::time::Duration;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = tui::App::new();
    let client = reqwest::Client::new();

    let result = run_app(&mut terminal, &mut app, &client).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {e}");
    }

    Ok(())
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut tui::App,
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| tui::draw(f, app))?;

        // Process distillation events
        app.process_events();

        // Start distillation if requested
        if app.running && app.screen == tui::Screen::Setup {
            app.screen = tui::Screen::Running;
            app.total_target = app.distill_config.n_samples;
            app.completed = 0;
            app.errors = 0;
            app.samples.clear();
            app.log.clear();
            app.progress = 0.0;

            let (tx, rx) = mpsc::unbounded_channel();
            app.rx = Some(rx);

            let client = client.clone();
            let api_cfg = app.api_config.clone();
            let dist_cfg = app.distill_config.clone();

            tokio::spawn(async move {
                // Phase 1: generate questions
                let questions = match distill::generate_questions(&client, &api_cfg, &dist_cfg, &tx).await {
                    Ok(q) => q,
                    Err(e) => {
                        let _ = tx.send(distill::ProgressEvent::Error(0, format!("Question generation failed: {e}")));
                        let _ = tx.send(distill::ProgressEvent::Done(0));
                        return;
                    }
                };

                // Phase 2: distill answers
                let samples = match distill::distill_answers(&client, &api_cfg, &dist_cfg, questions, tx.clone()).await {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(distill::ProgressEvent::Error(0, format!("Distillation failed: {e}")));
                        let _ = tx.send(distill::ProgressEvent::Done(0));
                        return;
                    }
                };

                // Phase 3: save
                match distill::save_dataset(&samples, &dist_cfg.output_format, &dist_cfg.output_path) {
                    Ok(n) => {
                        let _ = tx.send(distill::ProgressEvent::Done(n));
                    }
                    Err(e) => {
                        let _ = tx.send(distill::ProgressEvent::Error(0, format!("Save failed: {e}")));
                        let _ = tx.send(distill::ProgressEvent::Done(samples.len()));
                    }
                }
            });
        }

        // Poll events with timeout (non-blocking for async updates)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if app.handle_key(key.code) {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}
