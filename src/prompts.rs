//! Prompt store: kuni-master's `src/prompts.*` with live hot-reload.
//!
//! Seeds `prompts/{system,messages_epilogue,anti_repeat}.md` with the compiled-in
//! defaults, then polls the directory so the user can tune prompts without a restart.
//! `${CHARACTER_NAME}` / `${CHARACTER_NICKNAME}` / `${PAPIK_NAME}` placeholders are
//! substituted on every read, so name changes hot-reload too.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

const DEFAULT_SYSTEM: &str = include_str!("../prompts/system.md");
const DEFAULT_CHARACTER_BASE: &str = include_str!("../prompts/character_base.md");
const DEFAULT_CHARACTER_APPEARANCE: &str = include_str!("../prompts/character_appearance.md");
const DEFAULT_EPILOGUE: &str = include_str!("../prompts/messages_epilogue.md");
const DEFAULT_ANTI_REPEAT: &str = include_str!("../prompts/anti_repeat.md");

const RELOAD_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawPrompts {
    pub system: String,
    pub character_base: String,
    pub character_appearance: String,
    pub messages_epilogue: String,
    /// Shown to the LLM when it sends a repeated message (`prompts().antiRepeatPrompt`).
    pub anti_repeat: String,
}

/// Rendered prompts with the variables substituted (kuni's `Prompts` after `promptPrintf`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Prompts {
    pub system: String,
    pub character_base: String,
    pub character_appearance: String,
    pub messages_epilogue: String,
    pub anti_repeat: String,
}

#[derive(Debug, Clone, Default)]
pub struct PromptVars {
    pub character_name: String,
    pub character_nickname: String,
    pub papik_name: String,
}

pub struct PromptStore {
    dir: PathBuf,
    raw: RwLock<RawPrompts>,
}

impl PromptStore {
    /// The compiled-in anti-repeat prompt (kuni's `prompts().antiRepeatPrompt` default).
    pub fn default_anti_repeat() -> &'static str {
        Self::strip_header(DEFAULT_ANTI_REPEAT)
    }

    /// Seeds missing files with the compiled-in defaults and loads everything.
    pub async fn bootstrap(data_dir: &Path) -> anyhow::Result<Arc<Self>> {
        let dir = data_dir.to_path_buf();
        tokio::fs::create_dir_all(&dir).await?;
        Self::seed_once(&dir, "system.md", DEFAULT_SYSTEM).await?;
        Self::seed_once(&dir, "character_base.md", DEFAULT_CHARACTER_BASE).await?;
        Self::seed_once(&dir, "character_appearance.md", DEFAULT_CHARACTER_APPEARANCE).await?;
        Self::seed_once(&dir, "messages_epilogue.md", DEFAULT_EPILOGUE).await?;
        Self::seed_once(&dir, "anti_repeat.md", DEFAULT_ANTI_REPEAT).await?;

        let raw = Self::load_all(&dir).await?;
        let store = Arc::new(Self {
            dir,
            raw: RwLock::new(raw),
        });
        HotReloader::spawn(store.clone());
        Ok(store)
    }

    async fn seed_once(dir: &Path, name: &str, default: &str) -> anyhow::Result<()> {
        let path = dir.join(name);
        if !path.exists() {
            tokio::fs::write(path, default).await?;
            tracing::info!("seeded prompts/{name}");
        }
        Ok(())
    }

    /// kuni's `reg()`: strips the `---`-delimited header block (title/description),
    /// keeping only the body after the second `---`.
    fn strip_header(raw: &str) -> &str {
        let mut lines = raw.lines();
        match lines.next() {
            Some("---") => {}
            _ => return raw,
        }
        let mut count = 0;
        let mut offset = raw.len();
        for (i, line) in raw.lines().enumerate() {
            if line == "---" {
                count += 1;
                if count == 2 {
                    // Skip this "---" line and the newline that follows it.
                    offset = raw
                        .lines()
                        .take(i + 1)
                        .map(|l| l.len() + 1)
                        .sum();
                    break;
                }
            }
        }
        raw.get(offset..).unwrap_or(raw)
    }

    /// kuni's `promptPrintf`-like substitution of the template variables.
    fn substitute(raw: &str, vars: &PromptVars) -> String {
        raw.replace("${CHARACTER_NAME}", &vars.character_name)
            .replace("${CHARACTER_NICKNAME}", &vars.character_nickname)
            .replace("${PAPIK_NAME}", &vars.papik_name)
    }

    async fn load_all(dir: &Path) -> anyhow::Result<RawPrompts> {
        let read_file = |name: &'static str| async move { tokio::fs::read_to_string(dir.join(name)).await.ok() };
        let system_raw = read_file("system.md").await.unwrap_or_else(|| DEFAULT_SYSTEM.to_string());
        let base_raw = read_file("character_base.md")
            .await
            .unwrap_or_else(|| DEFAULT_CHARACTER_BASE.to_string());
        let appearance_raw = read_file("character_appearance.md")
            .await
            .unwrap_or_else(|| DEFAULT_CHARACTER_APPEARANCE.to_string());
        let epilogue_raw =
            read_file("messages_epilogue.md").await.unwrap_or_else(|| DEFAULT_EPILOGUE.to_string());
        let anti_raw = read_file("anti_repeat.md").await;
        Ok(RawPrompts {
            system: Self::strip_header(&system_raw).to_string(),
            character_base: Self::strip_header(&base_raw).to_string(),
            character_appearance: Self::strip_header(&appearance_raw).to_string(),
            messages_epilogue: Self::strip_header(&epilogue_raw).to_string(),
            anti_repeat: anti_raw
                .as_deref()
                .map(Self::strip_header)
                .map(str::to_string)
                .unwrap_or_else(|| Self::strip_header(DEFAULT_ANTI_REPEAT).to_string()),
        })
    }

    /// Renders the prompts with the current variable values.
    /// Mirrors kuni's `prompts()` accessor plus `AppBase::getSystemPrompt` variables.
    pub async fn get(&self, vars: &PromptVars) -> Prompts {
        let raw = self.raw.read().await.clone();
        Prompts {
            system: Self::substitute(&raw.system, vars),
            character_base: Self::substitute(&raw.character_base, vars),
            character_appearance: Self::substitute(&raw.character_appearance, vars),
            messages_epilogue: Self::substitute(&raw.messages_epilogue, vars),
            anti_repeat: Self::substitute(&raw.anti_repeat, vars),
        }
    }

    async fn maybe_reload(&self) -> bool {
        match Self::load_all(&self.dir).await {
            Ok(new) => {
                let mut raw = self.raw.write().await;
                if *raw != new {
                    *raw = new;
                    true
                } else {
                    false
                }
            }
            Err(e) => {
                tracing::warn!("prompts reload failed: {e}");
                false
            }
        }
    }
}

/// Simple polling hot-reload (the two files are a few KiB; mtimes are unreliable on
/// broad file systems, so content comparison is used to avoid churn).
struct HotReloader;

impl HotReloader {
    fn spawn(store: Arc<PromptStore>) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(RELOAD_INTERVAL).await;
                if store.maybe_reload().await {
                    tracing::info!("prompts hot-reloaded");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_header_block() {
        let raw = "---\ntitle\ndescription\n---\nbody line 1\nbody line 2\n";
        assert_eq!(PromptStore::strip_header(raw), "body line 1\nbody line 2\n");
        assert_eq!(PromptStore::strip_header("no header"), "no header");
    }

    #[test]
    fn substitutes_vars() {
        let vars = PromptVars {
            character_name: "Kuni".into(),
            character_nickname: "@kunii_chan".into(),
            papik_name: "Alex2772".into(),
        };
        let out = PromptStore::substitute("Hi ${CHARACTER_NAME} aka ${CHARACTER_NICKNAME}, ${PAPIK_NAME}!", &vars);
        assert_eq!(out, "Hi Kuni aka @kunii_chan, Alex2772!");
    }

    #[test]
    fn defaults_render() {
        assert!(DEFAULT_SYSTEM.contains("You are an AI that thinks you are a human"));
        assert!(DEFAULT_SYSTEM.contains("${CHARACTER_NICKNAME}"));
        assert!(DEFAULT_EPILOGUE.contains("reject playfully"));
        assert!(DEFAULT_CHARACTER_BASE.contains("When asked for your name"));
        assert!(DEFAULT_CHARACTER_APPEARANCE.contains("${CHARACTER_NAME}"));
        // Stripped body has no leftover header.
        assert!(!PromptStore::strip_header(DEFAULT_SYSTEM).starts_with("---"));
        assert!(!PromptStore::strip_header(DEFAULT_CHARACTER_BASE).starts_with("---"));
    }
}
