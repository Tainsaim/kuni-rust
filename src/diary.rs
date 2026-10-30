//! Diary: markdown-file-backed long-term memory with embedding-based queries.
//!
//! A port of kuni-master's `src/Diary.{h,cpp}`: each entry is a separate
//! `diary/<id>.md` file that may start with a `---`-delimited JSON metadata block
//! (`score`, `confidence`, `lastUsed`, `usageCount`, `embedding`). The entry list is
//! cached in memory; `save()` persists to disk *without* touching the cache, and
//! `reload()` invalidates the cache, both exactly like kuni.
//!
//! The free functions here are ports of kuni's `util/format_past_hours.h`,
//! `Worker.cpp contextEmbedding` and the pure-text parts of
//! `util/diary_save_entries.h` / `util/important_things_to_remember.h`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, MutexGuard};

use crate::config::ConfigHub;
use crate::llm::{Llm, Message};

pub struct Diary {
    dir: PathBuf,
    llm: Arc<Llm>,
    config: Arc<ConfigHub>,
    /// Lazily initialized cache of parsed entries (`aui::lazy<std::list<EntryEx>>`).
    /// `None` == invalidated (`reload()`), re-read from disk on next access.
    cache: AsyncMutex<Option<Vec<EntryEx>>>,
}

/// Simple representation of a diary entry (id + raw markdown).
pub struct Entry {
    pub id: String,
    pub text: String,
}

/// kuni's `EntryEx::Metadata` (serde ↔ the JSON block inside `---` fences).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Metadata {
    /// Similarity score used for ranking queries.
    pub score: f64,
    /// Confidence factor, ∈ {-1..1}: -1 lie, 0 theory/default,
    /// 1 ground truth (immutable by sleep).
    pub confidence: f32,
    /// Human-readable timestamp of the last access.
    pub last_used: String,
    /// Number of times the entry has been accessed.
    pub usage_count: i32,
    /// Embedding vector used for similarity calculations.
    pub embedding: Vec<f64>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            last_used: "never".to_string(),
            usage_count: 0,
            embedding: Vec::new(),
        }
    }
}

/// Extended entry: parsed metadata + freeform body (kuni's `EntryEx`).
#[derive(Clone, Debug)]
pub struct EntryEx {
    pub id: String,
    pub metadata: Metadata,
    pub freeform_body: String,
}

impl EntryEx {
    /// kuni's `EntryEx::incrementUsageCount`.
    pub fn increment_usage_count(&mut self) {
        self.metadata.usage_count += 1;
        self.metadata.last_used = format_now();
    }
}

/// Query result: an entry snapshot plus its normalized relatedness
/// (0 = unrelated, 1 = copypasta).
#[derive(Clone, Debug)]
pub struct EntryExAndRelatedness {
    pub entry: EntryEx,
    pub relatedness: f64,
}

/// kuni's `Diary::QueryOpts` (its `filter` is unused by the ported call sites).
#[derive(Clone, Copy, Debug)]
pub struct QueryOpts {
    /// How much confidence boosts the relatedness score.
    pub confidence_factor: f64,
    /// Hard cap of the result set.
    pub max_entry_count: usize,
}

impl Default for QueryOpts {
    fn default() -> Self {
        Self {
            confidence_factor: 0.01,
            max_entry_count: 10,
        }
    }
}

impl Diary {
    pub fn new(dir: PathBuf, llm: Arc<Llm>, config: Arc<ConfigHub>) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self {
            dir,
            llm,
            config,
            cache: AsyncMutex::new(None),
        }
    }

    // Guard over the raw cache slot, loading it from disk on demand.
    async fn cache_loaded(&self) -> MutexGuard<'_, Option<Vec<EntryEx>>> {
        let mut guard = self.cache.lock().await;
        if guard.is_none() {
            let start = std::time::Instant::now();
            let entries = Self::parse(Self::read(&self.dir).await);
            tracing::debug!(
                "Diary: loaded {} entries, took {:.2}s",
                entries.len(),
                start.elapsed().as_secs_f64()
            );
            *guard = Some(entries);
        }
        guard
    }

    /// Cached entries (kuni's `Diary::list`).
    pub async fn is_empty(&self) -> bool {
        self.cache_loaded().await.as_ref().unwrap().is_empty()
    }

    /// Invalidate the cached diary so that it will be reloaded on next access
    /// (kuni's `Diary::reload`, our `AUI_DEFER` in dump flows).
    pub async fn reload(&self) {
        *self.cache.lock().await = None;
    }

    /// Scans the diary dir for markdown files (`Diary::read`).
    async fn read(dir: &Path) -> Vec<Entry> {
        let mut out = Vec::new();
        let mut it = match tokio::fs::read_dir(dir).await {
            Ok(it) => it,
            Err(e) => {
                tracing::error!("Diary::read {}: {e}", dir.display());
                return out;
            }
        };
        while let Ok(Some(file)) = it.next_entry().await {
            let path = file.path();
            if path.is_dir() || path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|n| n.to_str()) else {
                continue;
            };
            match tokio::fs::read_to_string(&path).await {
                Ok(text) => out.push(Entry {
                    id: name.to_string(),
                    text,
                }),
                Err(e) => tracing::error!("Diary::read {:?}: {e}", path),
            }
        }
        out
    }

    /// kuni's `Diary::parse` over raw file entries.
    fn parse(diary: Vec<Entry>) -> Vec<EntryEx> {
        diary
            .into_iter()
            .map(|entry| {
                let text = entry.text.trim_matches('\n').to_string();
                parse_one(&entry.id, &text)
            })
            .collect()
    }

    /// Persists a raw entry to `<id>.md` (`Diary::save(Entry)`).
    pub async fn save_entry(&self, entry: &Entry) {
        let path = self.dir.join(format!("{}.md", entry.id));
        if let Err(e) = tokio::fs::write(&path, &entry.text).await {
            tracing::error!("Diary::save {}: {e}", path.display());
        }
    }

    /// Persists an extended entry with the `---metadata---` prologue
    /// (`Diary::save(EntryEx)`).
    pub async fn save(&self, entry: &EntryEx) {
        let metadata = match serde_json::to_string(&entry.metadata) {
            Ok(json) => json,
            Err(e) => {
                tracing::error!("Diary::save {}: metadata serialization: {e}", entry.id);
                return;
            }
        };
        self.save_entry(&Entry {
            id: entry.id.clone(),
            text: format!("---\n{metadata}\n---\n{}\n", entry.freeform_body),
        })
        .await
    }

    /// kuni's `Diary::query`: scores every entry against `query_vec` (embedding
    /// lazily), normalizes to [0..1], sorts by relatedness and keeps the top
    /// `maxEntryCount`. Returns owned snapshots.
    pub async fn query(
        &self,
        query_vec: &[f64],
        opts: QueryOpts,
    ) -> Result<Vec<EntryExAndRelatedness>, String> {
        let mut out = Vec::new();
        {
            let mut cache = self.cache_loaded().await;
            let entries = cache.as_mut().unwrap();
            for entry in entries.iter_mut() {
                let relatedness = self
                    .entry_is_related(entry, query_vec, opts)
                    .await
                    .map_err(|e| format!("query({}): {e}", entry.id))?;
                out.push(EntryExAndRelatedness {
                    entry: entry.clone(),
                    relatedness,
                });
            }
        }
        out.sort_by(|a, b| {
            b.relatedness
                .partial_cmp(&a.relatedness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(opts.max_entry_count);
        Ok(out)
    }

    /// kuni's `Diary::entryIsRelated`: generates and persists the embedding on miss.
    async fn entry_is_related(
        &self,
        entry: &mut EntryEx,
        context: &[f64],
        opts: QueryOpts,
    ) -> Result<f64, String> {
        if entry.freeform_body.is_empty() {
            return Ok(0.0);
        }
        if entry.metadata.embedding.len() != context.len() {
            let text = entry.freeform_body.replace('\t', "  ");
            entry.metadata.embedding = self
                .llm
                .embed(&text)
                .await
                .map_err(|_| format!("while generating embedding vector for {}", entry.id))?;
            self.save(entry).await;
        }
        Ok((crate::tg::human::cosine_similarity(context, &entry.metadata.embedding) + 1.0) / 2.0
            + entry.metadata.confidence as f64 * opts.confidence_factor)
    }

    /// Removes the id from the in-memory cache only (`Diary::unload`'s cache side;
    /// the caller persists the entry separately). Returns `true` when found.
    pub async fn forget(&self, id: &str) -> bool {
        let mut cache = self.cache_loaded().await;
        let entries = cache.as_mut().unwrap();
        let len_before = entries.len();
        entries.retain(|e| e.id != id);
        len_before != entries.len()
    }

    /// Replaces the cached entry with `entry` (kuni mutates cache entries in place
    /// when scoring them during ask's `queryDiary`).
    pub async fn retain(&self, entry: EntryEx) {
        let mut cache = self.cache_loaded().await;
        let entries = cache.as_mut().unwrap();
        match entries.iter_mut().find(|e| e.id == entry.id) {
            Some(cached) => *cached = entry,
            None => entries.push(entry),
        }
    }

    // =====================================================================
    // util/diary_save_entries.h (the plain-text part)
    // =====================================================================

    /// Splits an LLM summary by `---`, applies the plagiarism check and persists
    /// every non-duplicate piece (`util::diarySaveEntries`).
    pub async fn diary_save_entries(&self, summary: &str) -> Result<Vec<EntryEx>, String> {
        let mut id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|t| t.as_secs() as i64)
            .unwrap_or(0);

        // stupid AI sometimes messes up with separators
        let summary = summary.replace("- --", "---").replace("-- -", "---");
        let threshold = self.config.get().await.misc.diary_plagiarism_threshold as f64;

        let mut result = Vec::new();
        for take in summary.split("---") {
            if take.chars().count() < 20 {
                continue; // random shit
            }
            let embedding = self
                .llm
                .embed(take)
                .await
                .map_err(|e| format!("diary embed: {e}"))?;

            if let Some(first) = self
                .query(
                    &embedding,
                    QueryOpts {
                        confidence_factor: 0.0,
                        max_entry_count: 10,
                    },
                )
                .await?
                .first()
            {
                tracing::info!(
                    "diarySaveEntries {}.md: plagiarism factor other_id=\"{}\" relatedness={}",
                    id,
                    first.entry.id,
                    first.relatedness
                );
                if first.relatedness > threshold {
                    tracing::info!(
                        "diarySaveEntries {}.md: won't store because it's plagiarism other_id=\"{}\"",
                        id,
                        first.entry.id
                    );
                    continue;
                }
            }

            let entry = EntryEx {
                id: id.to_string(),
                metadata: Metadata {
                    embedding,
                    ..Default::default()
                },
                freeform_body: take.to_string(),
            };
            id += 1;
            self.save(&entry).await;
            result.push(entry);
        }
        Ok(result)
    }
}

/// kuni's `Diary::parse` lambda: extracts the `---json---` prologue; falls back to
/// the whole (trimmed) text with default metadata on failure.
fn parse_one(id: &str, text: &str) -> EntryEx {
    let fallback = |body: String| EntryEx {
        id: id.to_string(),
        metadata: Metadata::default(),
        freeform_body: body,
    };
    let text = text.trim_matches('\n').to_string();
    if !text.starts_with("---") {
        return fallback(text);
    }
    let Some(rel) = text[4..].find("---") else {
        return fallback(text);
    };
    let end = rel + 4;
    match serde_json::from_str::<Metadata>(text[4..end].trim()) {
        Ok(metadata) => EntryEx {
            id: id.to_string(),
            metadata,
            freeform_body: text
                .get(end + 4..)
                .map(|b| b.trim_start_matches('\n').to_string())
                .unwrap_or_default(),
        },
        Err(_) => {
            tracing::error!("Diary::parse can't parse {id}");
            fallback(text)
        }
    }
}

/// UTC timestamp in the `YYYY-MM-DD hh:mm:ss` form for `Metadata::last_used` and
/// `format_past_hours` (kuni uses `{}` on `std::chrono::system_clock::now()`).
pub fn format_unix_seconds(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = (secs % 86_400) as i64;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year,
        month,
        day,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub fn format_now() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| format_unix_seconds(t.as_secs()))
        .unwrap_or_else(|_| "epoch?".to_string())
}

/// kuni's `util::formatPastHours`: `<now-hours>-<now>`.
pub fn format_past_hours(hours: u64) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0);
    format!(
        "{}-{}",
        format_unix_seconds(secs.saturating_sub(hours * 3600)),
        format_unix_seconds(secs)
    )
}

/// Howard Hinnant's `civil_from_days`: unix days → (y, m, d).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// kuni's `Worker.cpp contextEmbedding`: one embedding over the last messages'
/// (reasoning_content +) content glued with `---` separators.
pub async fn context_embedding(llm: &Llm, messages: &[Message]) -> Result<Vec<f64>, String> {
    debug_assert!(!messages.is_empty());
    let mut base_prompt = String::new();
    for message in messages {
        if !message.reasoning_content.is_empty() {
            base_prompt += &message.reasoning_content;
            base_prompt += "\n\n";
        }
        base_prompt += &message.content;
        base_prompt += "\n\n---\n\n";
    }
    llm.embed(&base_prompt).await
}

/// kuni's `util::importantThingsToRemember` prompt building: the USER message used
/// to distill the "working memory" out of the session (the chat pump lives in
/// `app.rs`).
pub fn important_things_to_remember_prompt(
    character_name: &str,
    previous_working_memory: &str,
) -> String {
    let mut prompt = format!(
        "What are important things in timespan {} (3 days) you should remember?\n",
        format_past_hours(24 * 3)
    );
    prompt += "Do not attempt to make tool_calls or #ask. Your job is to summarize your current tasks and \
               revisit tasks from previous session.\n";
    if !previous_working_memory.is_empty() {
        prompt += "\nHere is the PREVIOUS <things_to_remember> from the last session. You MUST preserve ALL items verbatim from it, except:\n";
        prompt += "1. Completed tasks — mark them as done or remove\n";
        prompt += "2. Items that have NOT been updated for more than 3 days — you may forget them\n";
        prompt += "\nPrevious working memory:\n<previous_things_to_remember>\n";
        prompt += previous_working_memory;
        prompt += "\n</previous_things_to_remember>\n";
        prompt += "Important: the content of previous_things_to_remember will be overwritten by your next response. Make sure to preserve:";
        prompt += "- unfisnihed tasks (not older than 3 days);";
        prompt += "- reminders (not older that 3 days or without a deadline)\n";
    }
    prompt += "\nYou must include:\n- promises\n- reminders\n- unfinished tasks\n- responsibilities\n";
    prompt += &format!(
        "- {character_name}'s current emotional state: sad, arousal, betrayed, grief, sorrow, pensiveness, anger, annoyed,   fear, amazement, surprise, distracted, love, trust, accepted, joy, serenity, embarrassed, proud,   envious, jealous, grateful, disappointed, hopeful, lonely, nostalgic, relieved, resentful\n"
    );
    prompt += &format!(
        "- {character_name}'s current physical state: energetic, tired, exhausted, sleepy, restless, tense, relaxed,   hungry, thirsty, sore, dizzy, nauseous, feverish, cold, warm, comfortable, uncomfortable,   in_pain, injured, healthy, sick, weak, stiff, trembling, numb, wet, dry, stimulated\n"
    );
    prompt += "- other important details\n";
    prompt += "In your response, you must include previous <things_to_remember></things_to_remember>. DO NOT ALTER THEIR DESCRIPTION, thought you can expand them. You should not include older, outdated items and completed tasks.\n";
    prompt += "You must write briefly (100-500 words), structurize output, INCLUDE DATES.\n";
    prompt += "Each item MUST have a \"last updated\" date. Example format:\n";
    prompt += "- Напомнить Алексею про оплату хостинга до 15 мая — последнее обновление: Apr 28\n";
    prompt += "- Жду ответ от Марии по поводу встречи в пятницу — последнее обновление: Apr 29\n";
    prompt += "- Обещала скинуть рецепт пасты Коду — последнее обновление: Apr 27\n";
    prompt += "- Проверить статус заказа на Ozon — последнее обновление: Apr 25\n";
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metadata_block() {
        let raw = "---\n{\"confidence\":0.5}\n---\nbody\nmore body\n";
        let parsed = parse_one("1", raw);
        assert_eq!(parsed.freeform_body, "body\nmore body");
        assert!((parsed.metadata.confidence - 0.5f32).abs() < 1e-6);
    }

    #[test]
    fn parses_entry_without_metadata() {
        let parsed = parse_one("2", "just some text\n");
        assert_eq!(parsed.freeform_body, "just some text");
        assert_eq!(parsed.metadata.usage_count, 0);
        assert_eq!(parsed.metadata.last_used, "never");
        assert!(parsed.metadata.embedding.is_empty());
    }

    #[test]
    fn broken_json_falls_back_to_whole_text() {
        let parsed = parse_one("3", "---\n{not-a-json-json}\n---\nbody\n");
        // kuni's catch: metadata reset to defaults, body = whole (trimmed) text.
        assert_eq!(parsed.freeform_body, "---\n{not-a-json-json}\n---\nbody");
        assert_eq!(parsed.metadata.usage_count, 0);
    }

    #[test]
    fn roundtrip_save_parse() {
        let entry = EntryEx {
            id: "42".to_string(),
            metadata: Metadata {
                confidence: 0.7,
                usage_count: 3,
                last_used: format_now(),
                ..Default::default()
            },
            freeform_body: "piece body".to_string(),
        };
        let json = serde_json::to_string(&entry.metadata).unwrap();
        let text = format!("---\n{json}\n---\n{}\n", entry.freeform_body);
        let parsed = parse_one("42", &text);
        assert_eq!(parsed.freeform_body, "piece body");
        assert!((parsed.metadata.confidence - 0.7f32).abs() < 1e-6);
        assert_eq!(parsed.metadata.usage_count, 3);
    }

    #[test]
    fn format_unix_seconds_roundtrip() {
        assert_eq!(format_unix_seconds(0), "1970-01-01 00:00:00");
        assert_eq!(format_unix_seconds(951_782_400), "2000-02-29 00:00:00");
        assert_eq!(format_unix_seconds(1_795_000_000), "2026-11-18 11:06:40");
    }
}
