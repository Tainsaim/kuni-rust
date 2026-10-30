//! Port of kuni-master's `config.h`/`config.cpp` (minus photo/audio generation and the
//! proxy server) to Rust, with hot-reload.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use notify::Watcher as _;
use tokio::sync::{watch, RwLock};

/// kuni's `Config::SLEEP_MAX_TIME` (`std::chrono::hours(6)`): the hard cap of the
/// sleeping diary consolidation.
pub const SLEEP_MAX_TIME: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

/// Mirrors kuni's `Config::LockdownMode`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LockdownMode {
    /// public
    #[default]
    None,
    ContactsOnly,
    PapikOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Endpoint {
    pub base_url: String,
    pub bearer_key: String,
}

impl Default for Endpoint {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1/".into(),
            bearer_key: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EndpointAndModel {
    pub endpoint: Endpoint,
    pub model: String,
}

impl Default for EndpointAndModel {
    fn default() -> Self {
        Self {
            endpoint: Endpoint::default(),
            model: "deepseek-v4-flash".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct General {
    pub character_name: String,
    pub character_nickname: String,
    pub papik_name: String,
    pub papik_chat_id: i64,
    pub telegram_api_id: i32,
    pub telegram_api_hash: String,
    pub telegram_enabled: bool,
    pub llm: EndpointAndModel,
    pub embedding: EndpointAndModel,
    pub lockdown: LockdownMode,
}

impl Default for General {
    fn default() -> Self {
        Self {
            character_name: "Kuni".into(),
            character_nickname: "@kunii_chan".into(),
            papik_name: "Alex2772".into(),
            papik_chat_id: 625207005,
            telegram_api_id: 0,
            telegram_api_hash: String::new(),
            telegram_enabled: true,
            llm: EndpointAndModel::default(),
            embedding: EndpointAndModel {
                endpoint: Endpoint::default(),
                model: "qwen3-embedding".into(),
            },
            lockdown: LockdownMode::PapikOnly,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Misc {
    pub can_write_to_a_new_person: bool,
    pub wake_up_on_pinned_chat: bool,
    pub randomly_go_sleep: bool,
    pub tool_reminder_probability: f32,
    pub diary_token_count_trigger: usize,
    pub diary_injection_max_length: usize,
    pub diary_plagiarism_threshold: f32,
    pub diary_min_relatedness: f32,
    /// kuni's `capabilityWebSearch` (web module itself lands later).
    pub capability_web_search: bool,
    pub chat_max_history_length: usize,
    pub llm_temperature: Option<f32>,
    pub llm_top_p: Option<f32>,
    pub llm_top_k: Option<f32>,
    pub llm_min_p: Option<f32>,
    pub llm_presence_penalty: Option<f32>,
    pub llm_repetition_penalty: Option<f32>,
    pub anti_repeat_trigger_max: f32,
    pub anti_repeat_trigger_avg: f32,
    pub anti_repeat_max_history: usize,
    pub suggest_ignore_chance: f32,
    pub request_timeout_secs: u64,
    pub remind_use_ask: bool,
    pub typing_simulation_min_wpm: i32,
    pub typing_simulation_max_wpm: i32,
    pub check_chats_on_startup: bool,
    pub chat_notification_filter: LockdownMode,
    pub can_join_chats: bool,
    pub can_leave_chats: bool,
    pub worker_count: usize,
}

impl Default for Misc {
    fn default() -> Self {
        Self {
            can_write_to_a_new_person: false,
            wake_up_on_pinned_chat: false,
            randomly_go_sleep: true,
            tool_reminder_probability: 0.02,
            diary_token_count_trigger: 40_000,
            diary_injection_max_length: 0,
            diary_plagiarism_threshold: 0.97,
            diary_min_relatedness: 0.5,
            capability_web_search: false,
            chat_max_history_length: 2000,
            llm_temperature: Some(0.2),
            llm_top_p: None,
            llm_top_k: None,
            llm_min_p: None,
            llm_presence_penalty: None,
            llm_repetition_penalty: None,
            anti_repeat_trigger_max: 0.95,
            anti_repeat_trigger_avg: 0.85,
            anti_repeat_max_history: 32,
            suggest_ignore_chance: 0.1,
            request_timeout_secs: 30,
            remind_use_ask: true,
            typing_simulation_min_wpm: 120,
            typing_simulation_max_wpm: 150,
            check_chats_on_startup: true,
            chat_notification_filter: LockdownMode::None,
            can_join_chats: false,
            can_leave_chats: true,
            worker_count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VisionEndpoints {
    /// Captioning/understanding model (image -> description). Absent = media shows as `[photo]`.
    #[serde(default)]
    pub llm_image_to_text: Option<EndpointAndModel>,
}

impl Default for VisionEndpoints {
    fn default() -> Self {
        Self {
            llm_image_to_text: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HearingEndpoints {
    /// Speech-to-text model (voice/video note -> transcription). Absent = `[voice message]`.
    #[serde(default)]
    pub llm_audio_to_text: Option<EndpointAndModel>,
}

impl Default for HearingEndpoints {
    fn default() -> Self {
        Self {
            llm_audio_to_text: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Capabilities {
    pub vision: VisionEndpoints,
    pub hearing: HearingEndpoints,
    pub use_stickers: bool,
    pub web_search: WebSearch,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            vision: VisionEndpoints::default(),
            hearing: HearingEndpoints::default(),
            use_stickers: false,
            web_search: WebSearch::default(),
        }
    }
}

/// Web search via Ollama key, matching kuni's `capabilities.web_search`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WebSearch {
    pub enabled: bool,
    pub ollama_bearer_key: String,
}

impl Default for WebSearch {
    fn default() -> Self {
        Self {
            enabled: false,
            ollama_bearer_key: String::new(),
        }
    }
}

/// Flood-specific knowledge files of this instance (one flood chat).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Flood {
    /// Generalized flood context (glossary), e.g. english.md.
    pub context_file: String,
    /// Rules of the specific flood, e.g. rules.md.
    pub rules_file: String,
    /// Telegram dialog id (Bot API format) of the flood chat this knowledge belongs to.
    /// 0 == "applies to any chat".
    pub chat_id: i64,
}

impl Default for Flood {
    fn default() -> Self {
        Self {
            context_file: "english.md".into(),
            rules_file: "rules.md".into(),
            chat_id: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: General,
    pub misc: Misc,
    pub capabilities: Capabilities,
    pub flood: Flood,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: General::default(),
            misc: Misc::default(),
            capabilities: Capabilities::default(),
            flood: Flood::default(),
        }
    }
}

const DEFAULT_TOML: &str = r#"# kuni-rs runtime configuration (hot-reloaded: edit without restarting).

[general]
character_name = "Kuni"
character_nickname = "@kunii_chan"
papik_name = "Alex2772"              # your nickname — the instance owner
papik_chat_id = 625207005            # your Telegram user ID
telegram_api_id = 0                  # get from https://my.telegram.org
telegram_api_hash = ""               # get from https://my.telegram.org
telegram_enabled = true

# OpenAI-compatible endpoints.
[general.llm]
model = "deepseek-v4-flash"
[general.llm.endpoint]
base_url = "http://localhost:11434/v1/"
bearer_key = ""

[general.embedding]
model = "qwen3-embedding"
[general.embedding.endpoint]
base_url = "http://localhost:11434/v1/"
bearer_key = ""

# NONE = public; CONTACTS_ONLY; PAPIK_ONLY = only you.
lockdown = "PAPIK_ONLY"

[misc]
can_write_to_a_new_person = false
wake_up_on_pinned_chat = false
randomly_go_sleep = true
tool_reminder_probability = 0.02
diary_token_count_trigger = 40000    # dump context to the diary past this token count
diary_injection_max_length = 0       # 0 disables relevance-based diary injection
diary_plagiarism_threshold = 0.97
diary_min_relatedness = 0.5          # kuni default; feeds mRelevanceThreshold
# capability_web_search = false       # kuni's capabilityWebSearch (web module lands later)
chat_max_history_length = 2000
llm_temperature = 0.2
# llm_top_p / llm_top_k / llm_min_p / llm_presence_penalty / llm_repetition_penalty
anti_repeat_trigger_max = 0.95
anti_repeat_trigger_avg = 0.85
anti_repeat_max_history = 32
suggest_ignore_chance = 0.1
request_timeout_secs = 30
remind_use_ask = true
typing_simulation_min_wpm = 120
typing_simulation_max_wpm = 150
check_chats_on_startup = true
chat_notification_filter = "NONE"
can_join_chats = false
can_leave_chats = true
worker_count = 1

[capabilities]
use_stickers = false

[capabilities.web_search]
enabled = false
ollama_bearer_key = ""

[capabilities.vision]
# Uncomment and fill to enable viewing photos sent by people (requires a vision model):
# llm_image_to_text = { model = "qwen3.5:9b", endpoint = { base_url = "http://localhost:11434/v1/", bearer_key = "" } }

[capabilities.hearing]
# Uncomment and fill to enable transcribing voice notes (requires a whisper endpoint):
# llm_audio_to_text = { model = "base", endpoint = { base_url = "http://localhost:9000/v1/", bearer_key = "" } }

[flood]
context_file = "english.md"   # generalized flood context/glossary
rules_file = "rules.md"       # rules of the specific flood
chat_id = 0                   # Telegram dialog id this knowledge belongs to (0 = any)
"#;

/// Live, hot-reloadable view of `config.toml`.
pub struct ConfigHub {
    path: PathBuf,
    inner: RwLock<Config>,
    rx: watch::Receiver<u64>,
    generation: std::sync::atomic::AtomicU64,
}

impl ConfigHub {
    /// Loads `config.toml`; if it does not exist, seed it with a commented default and error out
    /// (same UX as kuni: first run asks the user to populate the file).
    pub async fn bootstrap(path: &Path) -> anyhow::Result<Arc<Self>> {
        if !path.exists() {
            tokio::fs::write(path, DEFAULT_TOML).await?;
            anyhow::bail!(
                "No config.toml found — a template has been written to {}. \
                 Fill in at least [general] (telegram_api_id/api_hash, papik) and run again.",
                path.display()
            );
        }
        let raw = tokio::fs::read_to_string(path).await?;
        let cfg: Config = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
        Self::validate(&cfg)?;

        let (tx, rx) = watch::channel(0u64);
        let hub = Arc::new(Self {
            path: path.to_path_buf(),
            inner: RwLock::new(cfg),
            rx,
            generation: std::sync::atomic::AtomicU64::new(0),
        });
        let hub_rx = tx.clone();

        // Hot-reload watcher.
        let watch_path = hub.path.clone();
        let watcher_hub = Arc::downgrade(&hub);
        tokio::spawn(async move {
            let (notify_tx, mut notify_rx) = tokio::sync::mpsc::unbounded_channel();
            let mut watcher = match notify::recommended_watcher(move |res| {
                let _ = notify_tx.send(res);
            }) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!("config watcher init failed: {e}");
                    return;
                }
            };
            if let Err(e) = watcher.watch(
                watch_path.parent().unwrap_or(Path::new(".")),
                notify::RecursiveMode::NonRecursive,
            ) {
                tracing::error!("config watch failed: {e}");
                return;
            }
            loop {
                match notify_rx.recv().await {
                    Some(Ok(event)) => {
                        if !event.paths.iter().any(|p| p == &watch_path) {
                            continue;
                        }
                        // Editors do several writes; debounce.
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        if notify_rx.try_recv().is_ok() {
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                            let _ = notify_rx.try_recv();
                        }
                        if let Some(hub) = watcher_hub.upgrade() {
                            match tokio::fs::read_to_string(&watch_path).await {
                                Ok(raw) => match toml::from_str::<Config>(&raw) {
                                    Ok(new) => {
                                        if let Err(e) = Self::validate(&new) {
                                            tracing::error!("config reload rejected: {e}");
                                            continue;
                                        }
                                        *hub.inner.write().await = new;
                                        let new_gen = hub
                                            .generation
                                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                                            + 1;
                                        let _ = hub_rx.send(new_gen);
                                        tracing::info!("config.toml reloaded");
                                    }
                                    Err(e) => tracing::error!("config reload parse failed: {e}"),
                                },
                                Err(e) => tracing::error!("config reload read failed: {e}"),
                            }
                        }
                    }
                    Some(Err(e)) => tracing::warn!("config watcher error: {e}"),
                    None => break,
                }
            }
        });

        Ok(hub)
    }

    fn validate(cfg: &Config) -> anyhow::Result<()> {
        if cfg.general.telegram_api_id == 0 {
            anyhow::bail!("telegram_api_id must be set (https://my.telegram.org)");
        }
        if cfg.general.telegram_api_hash.is_empty() {
            anyhow::bail!("telegram_api_hash must be set (https://my.telegram.org)");
        }
        if !(0.0..=0.98).contains(&cfg.misc.suggest_ignore_chance) {
            anyhow::bail!("suggest_ignore_chance must be in [0, 0.98]");
        }
        Ok(())
    }

    pub async fn get(&self) -> Config {
        self.inner.read().await.clone()
    }

    /// Reference-counted read for short-lived borrows inside a single await-free scope.
    pub async fn snapshot(&self) -> SnapshotGuard {
        SnapshotGuard {
            cfg: self.inner.read().await.clone(),
        }
    }

    /// Waits until the config changed since `generation`.
    pub async fn changed(&self, generation: u64) -> (Config, u64) {
        let mut rx = self.rx.clone();
        while *rx.borrow() <= generation {
            if rx.changed().await.is_err() {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
        let generation = *rx.borrow();
        (self.inner.read().await.clone(), generation)
    }

    pub async fn generation(&self) -> u64 {
        *self.rx.borrow()
    }
}

pub struct SnapshotGuard {
    cfg: Config,
}

impl std::ops::Deref for SnapshotGuard {
    type Target = Config;

    fn deref(&self) -> &Config {
        &self.cfg
    }
}
