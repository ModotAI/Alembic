use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ApiConfig {
    pub provider: Provider,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Provider {
    Claude,
    OpenAI,
    Groq,
    Cerebras,
    OpenRouter,
    Mistral,
    GoogleAI,
    Ollama,
    Local(String),
}

impl Provider {
    pub fn name(&self) -> &str {
        match self {
            Self::Claude => "Claude",
            Self::OpenAI => "OpenAI",
            Self::Groq => "Groq",
            Self::Cerebras => "Cerebras",
            Self::OpenRouter => "OpenRouter",
            Self::Mistral => "Mistral",
            Self::GoogleAI => "Google AI",
            Self::Ollama => "Ollama",
            Self::Local(_) => "Local",
        }
    }
    pub fn default_model(&self) -> &str {
        match self {
            Self::Claude => "claude-sonnet-4-6",
            Self::OpenAI => "gpt-4o-mini",
            Self::Groq => "llama-3.3-70b-versatile",
            Self::Cerebras => "llama-3.3-70b",
            Self::OpenRouter => "nvidia/nemotron-3-super-120b-a12b:free",
            Self::Mistral => "mistral-small-latest",
            Self::GoogleAI => "gemini-2.5-flash",
            Self::Ollama => "llama3.1:8b",
            Self::Local(_) => "default",
        }
    }
    pub fn base_url(&self) -> &str {
        match self {
            Self::Claude => "https://api.anthropic.com",
            Self::OpenAI => "https://api.openai.com",
            Self::Groq => "https://api.groq.com/openai",
            Self::Cerebras => "https://api.cerebras.ai",
            Self::OpenRouter => "https://openrouter.ai/api",
            Self::Mistral => "https://api.mistral.ai",
            Self::GoogleAI => "https://generativelanguage.googleapis.com",
            Self::Ollama => "http://localhost:11434",
            Self::Local(url) => url,
        }
    }
    pub fn all() -> Vec<Provider> {
        vec![Self::Groq, Self::OpenRouter, Self::Ollama, Self::Cerebras, Self::Mistral, Self::OpenAI, Self::Claude]
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAI,
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
            max_tokens: 1024,
            temperature: 0.7,
        }
    }
}

pub async fn call_api(
    client: &reqwest::Client,
    config: &ApiConfig,
    system: &str,
    prompt: &str,
) -> Result<String> {
    match &config.provider {
        Provider::Claude => call_claude(client, config, system, prompt).await,
        _ => call_openai_compat(client, config, system, prompt).await,
    }
}

async fn call_claude(
    client: &reqwest::Client,
    config: &ApiConfig,
    system: &str,
    prompt: &str,
) -> Result<String> {
    #[derive(Serialize)]
    struct Req {
        model: String,
        max_tokens: u32,
        temperature: f32,
        system: String,
        messages: Vec<Message>,
    }
    #[derive(Deserialize)]
    struct Resp {
        content: Vec<ContentBlock>,
    }
    #[derive(Deserialize)]
    struct ContentBlock {
        text: Option<String>,
    }

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&Req {
            model: config.model.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            system: system.into(),
            messages: vec![Message {
                role: "user".into(),
                content: prompt.into(),
            }],
        })
        .send()
        .await?
        .json::<Resp>()
        .await?;

    Ok(resp.content.into_iter()
        .filter_map(|b| b.text)
        .collect::<Vec<_>>()
        .join(""))
}

async fn call_openai_compat(
    client: &reqwest::Client,
    config: &ApiConfig,
    system: &str,
    prompt: &str,
) -> Result<String> {
    #[derive(Serialize)]
    struct Req {
        model: String,
        max_tokens: u32,
        temperature: f32,
        messages: Vec<Message>,
    }

    let url = format!("{}/v1/chat/completions", config.provider.base_url());

    let resp = client
        .post(&url)
        .bearer_auth(&config.api_key)
        .json(&Req {
            model: config.model.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            messages: vec![
                Message { role: "system".into(), content: system.into() },
                Message { role: "user".into(), content: prompt.into() },
            ],
        })
        .send()
        .await?;

    let status = resp.status();
    let body = resp.text().await?;

    if !status.is_success() {
        return Err(anyhow::anyhow!("API error {}: {}", status, &body[..body.len().min(200)]));
    }

    // Parse manualmente per tollerare campi extra
    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("JSON parse error: {e} — body: {}", &body[..body.len().min(200)]))?;

    json["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No content in response: {}", &body[..body.len().min(200)]))
}
