use crate::api::{call_api, ApiConfig, Message};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct DistillConfig {
    pub topics: Vec<String>,
    pub n_samples: usize,
    pub system_prompt: String,
    pub output_format: OutputFormat,
    pub output_path: PathBuf,
    pub parallel: usize,
    pub lang: String,
}

#[derive(Clone, Debug)]
pub enum OutputFormat {
    ChatML,
    Llama,
    Alpaca,
    Custom(String), // template con {user} e {assistant}
}

impl OutputFormat {
    pub fn name(&self) -> &str {
        match self {
            Self::ChatML => "ChatML",
            Self::Llama => "Llama",
            Self::Alpaca => "Alpaca",
            Self::Custom(_) => "Custom",
        }
    }
    pub fn format_pair(&self, question: &str, answer: &str) -> String {
        match self {
            Self::ChatML => format!(
                "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            ),
            Self::Llama => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
            ),
            Self::Alpaca => {
                serde_json::to_string(&serde_json::json!({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })).unwrap_or_default()
            }
            Self::Custom(tmpl) => tmpl
                .replace("{user}", question)
                .replace("{assistant}", answer),
        }
    }
    pub fn all() -> Vec<OutputFormat> {
        vec![Self::ChatML, Self::Llama, Self::Alpaca]
    }
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            topics: vec![
                "general knowledge".into(),
                "science".into(),
                "history".into(),
                "coding".into(),
                "math".into(),
            ],
            n_samples: 1000,
            system_prompt: "You are a helpful AI assistant.".into(),
            output_format: OutputFormat::ChatML,
            output_path: "dataset.jsonl".into(),
            parallel: 3,
            lang: "en".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub question: String,
    pub answer: String,
    pub topic: String,
    pub tokens_est: usize,
}

/// Evento di progresso inviato alla TUI
#[derive(Clone, Debug)]
pub enum ProgressEvent {
    Phase(String),
    QuestionGenerated(usize, String),    // index, question
    AnswerGenerated(usize, Sample),       // index, full sample
    Error(usize, String),
    Done(usize),                          // total completed
}

/// Genera domande sintetiche via LLM
pub async fn generate_questions(
    client: &reqwest::Client,
    api_cfg: &ApiConfig,
    dist_cfg: &DistillConfig,
    tx: &mpsc::UnboundedSender<ProgressEvent>,
) -> Result<Vec<(String, String)>> {
    let _ = tx.send(ProgressEvent::Phase("Generating questions...".into()));

    let per_topic = dist_cfg.n_samples / dist_cfg.topics.len().max(1);
    let batch_size = 10; // domande per chiamata API
    let mut all_questions: Vec<(String, String)> = Vec::new();

    for (t_idx, topic) in dist_cfg.topics.iter().enumerate() {
        let mut remaining = per_topic;

        while remaining > 0 {
            let n = remaining.min(batch_size);
            remaining -= n;

            let prompt = if dist_cfg.lang == "it" {
                format!(
                    "Genera esattamente {n} domande diverse su '{topic}'.\n\
                     Ogni domanda deve essere completa, in italiano, e richiedere una risposta breve.\n\
                     Scrivi una domanda per riga. Niente numerazione, niente prefissi.\n\
                     Esempio:\n\
                     Qual è il fiume più lungo d'Italia?\n\
                     Chi ha progettato la cupola del Duomo di Firenze?"
                )
            } else {
                format!(
                    "Generate exactly {n} diverse, complete questions about '{topic}' in {lang}.\n\
                     Each question must be a full, self-contained question.\n\
                     Output one question per line. No numbering, no prefixes.\n\
                     Example:\n\
                     What causes earthquakes and how are they measured?\n\
                     How does photosynthesis work in plants?",
                    lang = dist_cfg.lang
                )
            };

            let mut response = None;
            for attempt in 0..3u64 {
                match call_api(client, api_cfg, "You generate questions for AI training datasets.", &prompt).await {
                    Ok(r) => { response = Some(r); break; }
                    Err(e) => {
                        if format!("{e}").contains("429") {
                            let wait = (attempt + 1) * 20;
                            let _ = tx.send(ProgressEvent::Error(0, format!("Rate limited on '{topic}', waiting {wait}s...")));
                            tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
                        } else {
                            let _ = tx.send(ProgressEvent::Error(0, format!("Question gen failed for {topic}: {e}")));
                            break;
                        }
                    }
                }
            }

            if let Some(resp) = response {
                for line in resp.lines() {
                    let q = line.trim().to_string();
                    if q.len() > 10 && !q.starts_with('#') {
                        let idx = all_questions.len();
                        all_questions.push((topic.clone(), q.clone()));
                        let _ = tx.send(ProgressEvent::QuestionGenerated(idx, q));
                    }
                }
            }

            // Delay tra batch
            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        }

        let _ = tx.send(ProgressEvent::Phase(format!("Generated {} questions ({}/{})", all_questions.len(), t_idx+1, dist_cfg.topics.len())));
    }

    Ok(all_questions)
}

/// Distilla risposte — chiama l'API con rate limiting e retry
pub async fn distill_answers(
    client: &reqwest::Client,
    api_cfg: &ApiConfig,
    dist_cfg: &DistillConfig,
    questions: Vec<(String, String)>,
    tx: mpsc::UnboundedSender<ProgressEvent>,
) -> Result<Vec<Sample>> {
    let _ = tx.send(ProgressEvent::Phase("Distilling answers...".into()));

    let mut samples = Vec::new();
    let delay_ms = 2100; // 30 req/min = 1 ogni 2s, con margine

    for (idx, (topic, question)) in questions.iter().enumerate() {
        // Retry con backoff
        let mut answer = None;
        for attempt in 0..3 {
            match call_api(client, api_cfg, &dist_cfg.system_prompt, question).await {
                Ok(a) => { answer = Some(a); break; }
                Err(e) => {
                    let err_str = format!("{e}");
                    if err_str.contains("429") {
                        // Rate limited — aspetta e riprova
                        let wait = (attempt + 1) * 15; // 15s, 30s, 45s
                        let _ = tx.send(ProgressEvent::Error(idx, format!("Rate limited, waiting {wait}s...")));
                        tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
                    } else {
                        let _ = tx.send(ProgressEvent::Error(idx, format!("{e}")));
                        break;
                    }
                }
            }
        }

        if let Some(answer) = answer {
            let tokens_est = (question.len() + answer.len()) / 4;
            let sample = Sample {
                question: question.clone(),
                answer,
                topic: topic.clone(),
                tokens_est,
            };
            let _ = tx.send(ProgressEvent::AnswerGenerated(idx, sample.clone()));
            samples.push(sample);
        }

        // Delay tra richieste per rispettare rate limit
        if idx < questions.len() - 1 {
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
        }
    }

    let _ = tx.send(ProgressEvent::Done(samples.len()));
    Ok(samples)
}

/// Salva il dataset nel formato scelto
pub fn save_dataset(
    samples: &[Sample],
    format: &OutputFormat,
    path: &std::path::Path,
) -> Result<usize> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut count = 0;
    for s in samples {
        let entry = match format {
            OutputFormat::Alpaca => serde_json::json!({
                "instruction": s.question,
                "input": "",
                "output": s.answer,
                "topic": s.topic,
            }),
            OutputFormat::ChatML | OutputFormat::Llama => serde_json::json!({
                "text": format.format_pair(&s.question, &s.answer),
                "topic": s.topic,
            }),
            OutputFormat::Custom(_) => serde_json::json!({
                "text": format.format_pair(&s.question, &s.answer),
                "topic": s.topic,
            }),
        };
        writeln!(f, "{}", serde_json::to_string(&entry)?)?;
        count += 1;
    }
    Ok(count)
}
