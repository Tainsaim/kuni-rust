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
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{rng, Rng as _};
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
    pub async fn list(&self) -> Vec<EntryEx> {
        self.cache_loaded().await.as_ref().unwrap().clone()
    }

    /// Whether the diary has no entries (uses the same cache as `list`).
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

    // =====================================================================
    // kuni's Diary::sleepingConsolidation (phase 4)
    // =====================================================================

    /// kuni's `Diary::sleepingConsolidation`: emulates human sleeping behavior
    /// against the diary. `sleeping_consolidator_prompt` is
    /// `prompts().sleepConsolidator`; `deadline` bounds this invocation with the
    /// current sleep window; `abort` is the wake-up flag (checked between steps).
    ///
    /// Note: kuni-master triggers this only from the tests/debug UI; kuni-rs runs
    /// it during the random go-sleep window instead
    /// (`Worker::process_randomly_go_sleep`).
    pub async fn sleeping_consolidation(
        &self,
        sleeping_consolidator_prompt: &str,
        deadline: Option<std::time::Instant>,
        abort: Option<Arc<AtomicBool>>,
    ) {
        // We reached sleep max time. How many memory pieces did we cover? This
        // depends on the LLM's processing speed: both cloud deepseek-chat and
        // (qwen3.5:9b on an RTX4090) can process about 500 of Kuni's diary entries
        // per hour. Anyway, we don't need to process all entries, neither the
        // human brain does. This also means our algorithm does not require memory
        // decay (i.e., deleting old irrelevant entries): Kuni will remember
        // everything, but they would not spend LLM processing power on reflecting
        // about the same shit every time. There's a small chance Kuni will
        // remember old stuff, see the random target pick.
        let cancelled = |sleep_start: std::time::Instant,
                         deadline: Option<std::time::Instant>,
                         abort: &Option<Arc<AtomicBool>>| {
            if sleep_start.elapsed() >= crate::config::SLEEP_MAX_TIME {
                return true;
            }
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    return true;
                }
            }
            if let Some(abort) = abort {
                if abort.load(std::sync::atomic::Ordering::SeqCst) {
                    return true;
                }
            }
            false
        };

        'outer: for _ in 0..CONSOLIDATION_MAX_ROUNDS {
            // Fresh parse of the diary, sorted by id descending: recent first, old
            // last. Kuni assigns the same list into mCachedDiary.
            {
                let mut entries = Self::parse(Self::read(&self.dir).await);
                entries.sort_by(|a, b| b.id.cmp(&a.id));
                *self.cache.lock().await = Some(entries);
            }

            let mut id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|t| t.as_secs() as i64)
                .unwrap_or(0);
            let count = self.cache_loaded().await.as_ref().unwrap().len();
            let sleep_start = std::time::Instant::now();

            while !self.cache_loaded().await.as_ref().unwrap().is_empty() {
                // we'll remove considered diary entries from the cache and save
                // them back to disk.
                if cancelled(sleep_start, deadline, &abort) {
                    break 'outer;
                }

                let target = {
                    let mut cache = self.cache_loaded().await;
                    let entries = cache.as_mut().unwrap();
                    if entries.is_empty() {
                        break;
                    }
                    let idx = if rng().random::<f64>() < 0.8 {
                        // Likely branch: during your sleep it's likely you see the
                        // past couple of days. We will pick up the most recent
                        // target. Also, the LLM really likes to write new entries
                        // that mostly duplicate contents of those that were loaded
                        // into its context via RAG from context's embedding; AFAIK
                        // new details or reflections were added. Maybe it's not a
                        // bad thing; considering the fact that below we will find
                        // related entries as well and mix them together.
                        0
                    } else {
                        // Unlikely branch: pick a random target. In perspective,
                        // this gives more shuffled chunks, so each sleep
                        // consolidation slightly different chunks are compared.
                        // This gives uniform distribution of information and
                        // merging/splitting behavior.
                        rng().random_range(0..entries.len())
                    };
                    entries.remove(idx)
                };

                let mut target = target;
                if target.metadata.embedding.is_empty() {
                    match self.llm.embed(&target.freeform_body).await {
                        Ok(vec) => target.metadata.embedding = vec,
                        Err(e) => {
                            tracing::error!("sleepingConsolidation can't embed {}: {e}", target.id);
                            break 'outer;
                        }
                    }
                }

                // (The query may race with the workers: kuni is single-threaded
                // here; a duplicate take just wastes a couple of LLM tokens.)
                let results = loop {
                    match self
                        .query(
                            &target.metadata.embedding,
                            QueryOpts {
                                confidence_factor: 0.0, // we need just embedding relatedness
                                max_entry_count: 10,
                            },
                        )
                        .await
                    {
                        Ok(r) => break r,
                        Err(e) => {
                            // kuni's `goto tryAgain`; we add a pause to the retry
                            // (kuni busy-retries, we can't afford the spin).
                            tracing::error!("sleepingConsolidation can't query {e}");
                            tokio::time::sleep(std::time::Duration::from_secs(RETRY_PAUSE_SECS)).await;
                        }
                    }
                };

                let mut ids: Vec<String> = vec![target.id.clone()];
                let mut ids_to_remove: Vec<String> = Vec::new();
                let mut body = String::new();
                let trigger = self.config.get().await.misc.diary_token_count_trigger;
                for (i, item) in results.iter().enumerate() {
                    if body.chars().count() > trigger && i >= 2 {
                        break;
                    }
                    if !body.is_empty() {
                        body += "\n\n---\n\n";
                    }
                    body += &format!("{{\"confidence\":{}}}", item.entry.metadata.confidence);
                    body += &item.entry.freeform_body;
                    body += "\n";
                    ids.push(item.entry.id.clone());
                    if item.entry.metadata.confidence < 0.9999999 {
                        ids_to_remove.push(item.entry.id.clone());
                    }
                    self.forget(&item.entry.id).await;
                }
                tracing::info!(
                    "[{}/{}] sleepingConsolidation: [{}]",
                    count - self.cache_loaded().await.as_ref().unwrap().len(),
                    count,
                    ids.join(", ")
                );
                tracing::debug!("sleepingConsolidation prompt: {body}");

                // kuni label `naxyi`: send the consolidator body and parse the
                // response; any LLM/parse error re-generates the whole chunk. The
                // parse cap protects against a pathological LLM burning tokens
                // forever (kuni retries indefinitely).
                let mut parse_attempts = 0u32;
                'naxyi: loop {
                    let messages = [crate::llm::Message::user(body.clone())];
                    let response = loop {
                        match self
                            .llm
                            .chat_streaming(Some(sleeping_consolidator_prompt), &messages, None, None)
                            .await
                        {
                            Ok((message, _)) => {
                                tracing::debug!(
                                    "sleepingConsolidation response: {}",
                                    message.content
                                );
                                break message.content;
                            }
                            Err(e) => {
                                tracing::error!("sleepingConsolidation can't chat {e}");
                                tokio::time::sleep(std::time::Duration::from_secs(RETRY_PAUSE_SECS))
                                    .await;
                            }
                        }
                    };

                    match self.parse_sleeping_response(&response, &mut id).await {
                        Ok(()) => break,
                        Err(e) => {
                            tracing::error!("sleepingConsolidation can't parse: {e}");
                            parse_attempts += 1;
                            if parse_attempts >= MAX_PARSE_ATTEMPTS {
                                break 'outer;
                            }
                            tokio::time::sleep(std::time::Duration::from_secs(RETRY_PAUSE_SECS)).await;
                            continue 'naxyi;
                        }
                    }
                }

                // since we have written refined diary entries, we don't need the
                // old ones.
                for id_to_remove in &ids_to_remove {
                    let file = self.dir.join(format!("{id_to_remove}.md"));
                    match tokio::fs::remove_file(&file).await {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => tracing::error!(
                            "sleepingConsolidation can't remove {}: {e}",
                            file.display()
                        ),
                    }
                }
            }
        }
        self.reload().await;
    }

    /// kuni's response split: `{"confidence":x}body` pieces separated by `\n---`.
    /// Saves the parsed pieces as it goes (kuni does the same, so a later parse
    /// failure leaves the already-parsed pieces saved and re-generates the rest).
    async fn parse_sleeping_response(&self, response: &str, next_id: &mut i64) -> Result<(), String> {
        for piece in response.split("\n---") {
            if piece.chars().count() < 10 {
                continue; // unknown shit
            }
            let Some(brace) = piece.find('}') else {
                return Err("no metadata block found".to_string());
            };
            #[derive(serde::Deserialize)]
            struct SleepMeta {
                confidence: f32,
            }
            let meta: SleepMeta =
                serde_json::from_str(&piece[..=brace]).map_err(|e| format!("metadata parse: {e}"))?;
            let freeform_body = piece[brace + 1..].trim_matches('\n');

            if meta.confidence < -0.99999 {
                // drop.
                continue;
            }

            // IMPORTANT CONSIDERATION: we'll delete old entries, replacing them
            // with the newer ones. This will effectively update their ids, thus,
            // during the next sleep, they will appear again closer to the
            // beginning of the processing queue. This algorithm encourages keeping
            // in mind the most recent events and reflecting on them specifically.
            self.save(&EntryEx {
                id: next_id.to_string(),
                metadata: Metadata {
                    confidence: meta.confidence.clamp(-0.99, 0.99),
                    ..Default::default()
                },
                freeform_body: freeform_body.to_string(),
            })
            .await;
            *next_id += 1;
        }
        Ok(())
    }
}

/// Cap for the kuni `goto naxyi` re-generation loop (kuni retries forever).
const MAX_PARSE_ATTEMPTS: u32 = 3;
/// Our pause between kuni's infinite retry loops (kuni busy-retries): a down LLM
/// endpoint can't spin the consolidation.
const RETRY_PAUSE_SECS: u64 = 5;
/// Safety bound of the consolidation outer loop; the real termination is the
/// deadline / wake-up flag checks.
const CONSOLIDATION_MAX_ROUNDS: usize = 8192;


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
