//! Worker: the autonomous notification-processing loop.
//! Port of kuni's `src/Worker.cpp` including the diary injection
//! (`naxyi_populate_ctx`), the `diaryDumpMessages` triggers, the random go-sleep
//! window (which runs the diary sleepingConsolidation) and the
//! actProactively-aware escape tools.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use rand::Rng as _;

use crate::app::App;
use crate::diary::{self, QueryOpts};
use crate::llm::{self, Message};
use crate::notifications::{Notification, RunningWorker};
use crate::tools::{self, Ctx, OpenAITools};

/// A long-lived worker: owns its conversation context (kuni's `mTemporaryContext`),
/// the "last seen message id" map and the shared tool state.
pub struct Worker {
    app: Arc<App>,
    session: Arc<StdMutex<Vec<Message>>>,
    seen: Arc<StdMutex<BTreeMap<i64, i32>>>,
    ask_called: Arc<AtomicBool>,
    tools: Arc<StdMutex<BTreeMap<String, tools::Tool>>>,
    /// The `pause`/`wait` escape tools' flag, reset per notification.
    pause_flag: Arc<AtomicBool>,
    /// kuni's `mWakeUp`: true when the sleep window should end early (papik
    /// messages call `App::wake_up_if_sleeping`).
    wake_up: Arc<AtomicBool>,
    /// kuni's `mRelevanceThreshold` (learned while injecting diary entries).
    relevance_threshold: f64,
}

impl Worker {
    fn new(app: Arc<App>, relevance_threshold: f64) -> Self {
        Self {
            app,
            session: Arc::new(StdMutex::new(Vec::new())),
            seen: Arc::new(StdMutex::new(BTreeMap::new())),
            ask_called: Arc::new(AtomicBool::new(false)),
            tools: Arc::new(StdMutex::new(BTreeMap::new())),
            pause_flag: Arc::new(AtomicBool::new(false)),
            wake_up: Arc::new(AtomicBool::new(false)),
            relevance_threshold,
        }
    }

    fn ctx(&self) -> Ctx {
        Ctx {
            app: self.app.clone(),
            args: serde_json::json!({}),
            all_tool_calls: Arc::new(Vec::new()),
            ask_called: self.ask_called.clone(),
            tools: self.tools.clone(),
            seen: self.seen.clone(),
            session: self.session.clone(),
        }
    }

    fn tool_set(&self) -> OpenAITools {
        OpenAITools {
            tools: self.tools.clone(),
        }
    }

    /// kuni's `Worker::updateTools`: merges the global tool set into the current
    /// tool map (upsert).
    async fn update_tools(&self) {
        let globals = tools::build_desktop_tool_list(&self.app, &self.ctx()).await;
        let mut guard = self.tools.lock().unwrap();
        for tool in globals {
            guard.insert(tool.name.clone(), tool);
        }
    }

    /// kuni's `processRandomlyGoSleep`: 1% chance to go afk for 15–120 minutes
    /// before processing the notification (it's humane + saves resources). The
    /// wake-up flag ends the window early. While Kuni sleeps, the diary
    /// sleepingConsolidation runs (kuni-master triggers it from the debug UI
    /// instead; a headless kernel sleeps → brain maintenance).
    async fn process_randomly_go_sleep(&self) {
        if !self.app.config.get().await.misc.randomly_go_sleep {
            return;
        }
        if rand::rng().random::<f64>() >= 0.01 {
            return;
        }
        let duration_mins = rand::rng().random_range(15..=120i64);
        tracing::info!("Going to sleep for {duration_mins} minutes");
        self.wake_up.store(false, Ordering::SeqCst);
        let duration = std::time::Duration::from_secs(duration_mins as u64 * 60);
        let deadline = std::time::Instant::now() + duration;
        let wake_up = self.wake_up.clone();
        let consolidator = {
            let diary = self.app.diary.clone();
            let abort = wake_up.clone();
            let prompt = self
                .app
                .prompts
                .get(&self.app.prompt_vars().await)
                .await
                .sleep_consolidator;
            async move {
                diary
                    .sleeping_consolidation(&prompt, Some(deadline), Some(abort))
                    .await;
            }
        };

        // Consolidation is brain maintenance performed *during* sleep; finishing
        // it early (for example, because the diary is empty) must not end the
        // 15–120 minute AFK window. Keep waiting until the deadline or papik's
        // wake-up signal while allowing consolidation to run in parallel.
        let mut consolidator = Box::pin(consolidator);
        let mut consolidation_done = false;
        while std::time::Instant::now() < deadline && !wake_up.load(Ordering::SeqCst) {
            let tick = deadline
                .saturating_duration_since(std::time::Instant::now())
                .min(std::time::Duration::from_secs(1));
            if consolidation_done {
                tokio::time::sleep(tick).await;
            } else {
                tokio::select! {
                    _ = &mut consolidator => consolidation_done = true,
                    _ = tokio::time::sleep(tick) => {}
                }
            }
        }
        if wake_up.load(Ordering::SeqCst) {
            tracing::info!("Early wake up");
        }
        // The consolidation may have been dropped mid-run (early wake): make sure
        // the cache is re-read on next access.
        self.app.diary.reload().await;
    }

    /// kuni's `processShortcutOpen`: a notification with a single action tool
    /// (typically `open`) has that action executed first, turning the bare
    /// notification into the action's full output before the LLM sees it.
    async fn process_shortcut_open(&self, notification: &mut Notification) {
        let only = {
            let guard = notification.actions.tools.lock().unwrap();
            if guard.len() == 1 {
                guard.values().next().map(|t| t.handler.clone())
            } else {
                None
            }
        };
        let Some(handler) = only else {
            return;
        };
        let result = handler(self.ctx())
            .await
            .unwrap_or_else(|e| format!("error while executing shortcut action: {e}"));
        notification.message = result;
    }

    /// The typing indicator callback (kuni's `onResponseAssembling`): pokes
    /// `chatActionTyping` at the currently opened chat.
    fn typing_cb(&self) -> llm::TypingCb {
        let app = self.app.clone();
        Arc::new(move || {
            let app = app.clone();
            Box::pin(async move {
                let chat_id = app
                    .currently_opened_chat
                    .load(std::sync::atomic::Ordering::SeqCst);
                if chat_id == 0 {
                    return;
                }
                use grammers_tl_types as tl;
                let action = tl::enums::SendMessageAction::SendMessageTypingAction;
                if let Err(e) = app.tg.send_action(chat_id, action).await {
                    tracing::debug!("typing indicator failed: {e}");
                }
            })
        })
    }

    /// kuni's `Worker::handleNotification`.
    async fn handle_notification(&mut self, notification: Notification) -> Result<(), String> {
        self.process_randomly_go_sleep().await;
        self.ask_called.store(false, Ordering::SeqCst);
        self.pause_flag.store(false, Ordering::SeqCst);

        // kuni's onStartedProcessing/onProcessed handlers on mActingProactively.
        let acting = notification.pin.as_deref() == Some(crate::app::PROACTIVE_PIN);
        if acting {
            self.app
                .acting_proactively
                .store(true, Ordering::SeqCst);
        }
        let result = self.handle_notification_inner(notification).await;
        if acting {
            self.app
                .acting_proactively
                .store(false, Ordering::SeqCst);
        }
        result
    }

    async fn handle_notification_inner(
        &mut self,
        mut notification: Notification,
    ) -> Result<(), String> {
        {
            let seeded = notification.actions.tools.lock().unwrap().clone();
            *self.tools.lock().unwrap() = seeded;
        }

        self.process_shortcut_open(&mut notification).await;
        tracing::info!("Processing notification: {}", notification.message);
        self.session
            .lock()
            .unwrap()
            .push(Message::user(std::mem::take(&mut notification.message)));

        let system_prompt = self.app.get_system_prompt().await;
        let cfg_remind = self.app.config.get().await.misc.remind_use_ask;

        loop {
            // kuni's naxyi_populate_ctx:
            self.populate_diary_context().await;

            // kuni's naxyi_preserve_ctx:
            self.update_tools().await;

            // escape tools (kuni inserts them right before the LLM call). While
            // acting proactively, a wait asks Kuni to keep trying to write someone.
            let nudge = Some((self.app.clone(), self.app.acting_proactively.clone()));
            {
                let mut guard = self.tools.lock().unwrap();
                guard.insert(
                    "pause".to_string(),
                    tools::escape_tool("pause", "Pauses the conversation", self.pause_flag.clone(), nudge.clone()),
                );
                guard.insert(
                    "wait".to_string(),
                    tools::escape_tool(
                        "wait",
                        "Wait until further notifications",
                        self.pause_flag.clone(),
                        nudge.clone(),
                    ),
                );
            }

            // remind LLM to call #ask before responding (incl. right before the
            // very first turn of this notification).
            if !self.ask_called.load(Ordering::SeqCst) && cfg_remind {
                let mut session = self.session.lock().unwrap();
                if let Some(last) = session.last_mut() {
                    last.content += "\n[system] Have you called #ask yet this turn? If the message involves personal topics, past events, or people you know — call #ask BEFORE send_telegram_message.";
                }
            }

            let tools_json = self.tool_set().as_json();
            let snapshot: Vec<Message> = self.session.lock().unwrap().clone();
            let (bot, usage) = self
                .app
                .llm
                .chat_streaming(
                    Some(&system_prompt),
                    &snapshot,
                    Some(&tools_json),
                    Some(self.typing_cb()),
                )
                .await
                .map_err(|e| format!("LLM call failed: {e}"))?;

            if bot.tool_calls.is_empty() {
                // no tool calls: each LLM turn should end with `wait` or `pause`.
                // guiderails to make LLM tool-centric (kuni's three scenarios).
                tracing::warn!("LLM didn't perform any action.");
                self.push_guiderail(&bot.content);
                continue;
            }

            // kuni's processIgnoreChance: with a small probability pretend the tool
            // calls failed with "do you really want to continue?" so the LLM
            // reconsiders (wait/pause calls escape begging).
            if self.should_beg_ignore().await
                && !bot
                    .tool_calls
                    .iter()
                    .any(|tc| tc.name == "wait" || tc.name == "pause")
            {
                let mut session = self.session.lock().unwrap();
                for tc in &bot.tool_calls {
                    session.push(Message::tool(
                        tc.id.clone(),
                        format!(
                            "Error: do you really want to continue? Think again; repeat `{}` to continue or call wait() to finish.",
                            tc.name
                        ),
                    ));
                }
                tracing::info!("Begging LLM to be lazy (ignore message)");
                continue;
            }

            // Execute the tool calls, push the assistant message and the `tool`
            // results into the context, then sprinkle kuni's planning hints onto
            // the last tool result.
            let did_send = bot
                .tool_calls
                .iter()
                .any(|tc| tc.name == "send_telegram_message");
            let results = self
                .tool_set()
                .handle_tool_calls(&self.ctx(), &bot.tool_calls)
                .await;
            {
                let mut session = self.session.lock().unwrap();
                session.push(bot);
                let mut tool_messages: Vec<Message> = results
                    .into_iter()
                    .map(|r| Message::tool(r.tool_call_id, r.content))
                    .collect();

                if let Some(last) = tool_messages.last_mut() {
                    let names = self.tool_set().names().join(", ");
                    last.content += &format!(
                        "\nWhat's your next action? Use a `tool` to act. Use #ask to consult with your knowledge database. The following tools available: {names}"
                    );
                    // if LLM sent a message without ever calling #ask this turn,
                    // inject the standard nag.
                    if did_send && cfg_remind {
                        last.content += "\n[system] Note: you sent a message without consulting #ask this turn. Next time, call #ask before send_telegram_message to enrich your response with memories and context.";
                    }
                }

                session.extend(tool_messages);
            }

            if self.pause_flag.load(Ordering::SeqCst) {
                // kuni's `finish` label: dump the context to the diary past the
                // token trigger.
                let trigger = self.app.config.get().await.misc.diary_token_count_trigger;
                if usage.total_tokens() >= trigger as u64 {
                    if let Err(e) = App::diary_dump_messages(&self.app, &self.session).await {
                        tracing::error!("diaryDumpMessages failed: {e}");
                    }
                }
                return Ok(());
            }
        }
    }

    /// Guiderails to make LLM tool-centric (kuni's three scenarios).
    fn push_guiderail(&self, content: &str) {
        let message = if content.contains("#send_telegram_message") {
            // qwen3.5 bug: misused examples
            "Nice thoughts! However you should be tool-centric. Make sure you made tool calls. The message you provided is not visible to anyone but you."
        } else {
            // gemma4 bug: replies with <message> xml instead of making tool calls
            // (kuni's both other scenarios share the same message).
            "Nice thoughts! However you should be tool-centric. Make sure you made tool calls. The message you provided is not visible to anyone but you. Call #wait if you are unsure."
        };
        self.session
            .lock()
            .unwrap()
            .push(Message::user(message.to_string()));
    }

    /// Was this turn probably born to perform an LLM lazy-ignore? kuni's
    /// `suggestIgnoreChance` config knob.
    async fn should_beg_ignore(&self) -> bool {
        let chance = self.app.config.get().await.misc.suggest_ignore_chance;
        chance > 0.0 && rand::rng().random::<f32>() < chance
    }

    /// kuni's `Worker::populateDiaryContext` (the `naxyi_populate_ctx` label):
    /// embeds the current conversation cue, picks the most related diary entries
    /// and inlines them into the last message. Learns `relevance_threshold`.
    async fn populate_diary_context(&mut self) {
        if self
            .app
            .config
            .get()
            .await
            .misc
            .diary_injection_max_length
            == 0
        {
            return;
        }
        if self.app.diary.is_empty().await {
            return;
        }

        // performs scan on diary based on context; this will find common cues which
        // are related to current conversation (contextEmbedding over last 3 messages).
        if self.session.lock().unwrap().is_empty() {
            return;
        }
        let last_messages: Vec<Message> = {
            let session = self.session.lock().unwrap();
            let skip = session.len().saturating_sub(3);
            session[skip..].to_vec()
        };
        let current_context =
            match diary::context_embedding(&self.app.llm, &last_messages).await {
                Ok(vec) => vec,
                Err(e) => {
                    tracing::error!("contextEmbedding: {e}");
                    return;
                }
            };
        let relatednesses = match self
            .app
            .diary
            .query(
                &current_context,
                QueryOpts {
                    confidence_factor: 0.0,
                    max_entry_count: 10,
                },
            )
            .await
        {
            Ok(list) => list,
            Err(e) => {
                tracing::error!("diary query: {e}");
                return;
            }
        };

        let mut diary_inject = String::new();
        for item in &relatednesses {
            if item.relatedness < self.relevance_threshold {
                if diary_inject.is_empty() {
                    // relax threshold for future queries (glm::mix(0.5, rel, 0.9)).
                    self.relevance_threshold = 0.5 * 0.1 + item.relatedness * 0.9;
                }
                break;
            }
            if diary_inject.chars().count() >= self.app.config.get().await.misc.diary_injection_max_length {
                // set the minimum constraint for the future queries
                self.relevance_threshold = item.relatedness;
                break;
            }
            match self.take_diary_entry(item).await {
                Some(s) => diary_inject += &s,
                None => {}
            }
        }

        if !diary_inject.is_empty() {
            if let Some(last) = self.session.lock().unwrap().last_mut() {
                let content = std::mem::take(&mut last.content);
                diary_inject += &content;
                last.content = diary_inject;
            }
        }
    }

    /// kuni's `Worker::takeDiaryEntry`: skips entries already present in context
    /// verbatim; otherwise bumps the entry score/usage and unloads it (the file
    /// stays on disk, it reappears after the next `reload()`).
    async fn take_diary_entry(
        &self,
        item: &diary::EntryExAndRelatedness,
    ) -> Option<String> {
        tracing::trace!("takeDiaryEntry: {}", item.entry.id);
        {
            let session = self.session.lock().unwrap();
            if session.iter().any(|m| m.content.contains(&item.entry.freeform_body)) {
                // if session already contains this diary entry verbatim - we don't
                // need to reinclude it - it makes no sense to consume tokens for the
                // same thing. The copypasted diary entry would not receive score.
                return None;
            }
        }

        let mut entry = item.entry.clone();
        entry.metadata.score += (item.relatedness - 0.5) * 2.0;
        entry.increment_usage_count();
        tracing::info!(
            "Loaded into context: {}.md relatedness={}\n{}",
            entry.id,
            item.relatedness,
            entry.freeform_body
        );
        let formatted_tag = "your_diary_page additional_context just_for_reasoning no_plagiarism no_copy";
        let result = format!("<{formatted_tag}>\n{}\n</{formatted_tag}>\n", entry.freeform_body);
        self.app.diary.save(&entry).await;
        self.app.diary.forget(&entry.id).await; // unload: drop from the cache
        Some(result)
    }

    fn clear_session(&self) {
        self.session.lock().unwrap().clear();
    }

    /// kuni's AUI_DEFER onOffline when a notification is fully processed: close the
    /// chat and drop the online status.
    fn after_notification(&self) {
        self.app
            .currently_opened_chat
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A registered worker: consumes notifications it is allowed to process.
pub struct WorkerRunner {
    worker: Worker,
    runner: RunningWorker,
}

impl WorkerRunner {
    /// The infinite processing loop (kuni's Worker::Worker enqueue + run).
    pub async fn loop_forever(mut self) {
        while let Some(notification) = self.runner.next().await {
            match self.worker.handle_notification(notification).await {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!("Failed to process notification: \"{e}\"");
                    // If there's a JSON error, it means we have irreversibly damaged
                    // context: the only thing to do is to drop the context entirely.
                    if e.to_lowercase().contains("json") {
                        tracing::warn!("Context is damaged. Dropping context");
                        self.worker.clear_session();
                    }
                }
            }
            self.worker.after_notification();
        }
    }
}

/// Spawns `count` workers for the app (kuni's `workerCount` loop in main.cpp).
pub async fn spawn_workers(app: &Arc<App>) {
    let cfg = app.config.get().await;
    let count = cfg.misc.worker_count.max(1);
    let relevance_threshold = cfg.misc.diary_min_relatedness as f64;
    for _ in 0..count {
        let runner = app.notifications.run().await;
        let worker = Worker::new(app.clone(), relevance_threshold);
        app.worker_sessions.lock().unwrap().push(worker.session.clone());
        app.worker_wakeups.lock().unwrap().push(worker.wake_up.clone());
        tokio::spawn(WorkerRunner { worker, runner }.loop_forever());
    }
}
