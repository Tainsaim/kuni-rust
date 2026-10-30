//! Worker: the autonomous notification-processing loop.
//! Port of kuni's `src/Worker.cpp` (minus the diary buffers, which ship in phase 3,
//! and the sleep machinery in phase 4).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use rand::Rng as _;

use crate::app::App;
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
}

impl Worker {
    fn new(app: Arc<App>) -> Self {
        Self {
            app,
            session: Arc::new(StdMutex::new(Vec::new())),
            seen: Arc::new(StdMutex::new(BTreeMap::new())),
            ask_called: Arc::new(AtomicBool::new(false)),
            tools: Arc::new(StdMutex::new(BTreeMap::new())),
            pause_flag: Arc::new(AtomicBool::new(false)),
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
                let action =
                    tl::enums::SendMessageAction::SendMessageTypingAction(
                        tl::types::SendMessageTypingAction {},
                    );
                if let Err(e) = app.tg.send_action(chat_id, action).await {
                    tracing::debug!("typing indicator failed: {e}");
                }
            })
        })
    }

    /// kuni's `Worker::handleNotification`.
    async fn handle_notification(&self, mut notification: Notification) -> Result<(), String> {
        self.ask_called.store(false, Ordering::SeqCst);
        self.pause_flag.store(false, Ordering::SeqCst);
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
            // (naxyi_populate_ctx: diary injection arrives in phase 3.)

            // kuni's naxyi_preserve_ctx:
            self.update_tools().await;

            // escape tools (kuni inserts them right before the LLM call).
            {
                let mut guard = self.tools.lock().unwrap();
                guard.insert(
                    "pause".to_string(),
                    tools::escape_tool(
                        "pause",
                        "Pauses the conversation",
                        self.pause_flag.clone(),
                    ),
                );
                guard.insert(
                    "wait".to_string(),
                    tools::escape_tool(
                        "wait",
                        "Wait until further notifications",
                        self.pause_flag.clone(),
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
            // (the context trimming trigger lands in phase 3: diaryDumpMessages.)
            let snapshot: Vec<Message> = self.session.lock().unwrap().clone();
            let (bot, _usage) = self
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
                // kuni's diaryDumpMessages check goes in phase 3.
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
    let count = app.config.get().await.misc.worker_count.max(1);
    for _ in 0..count {
        let runner = app.notifications.run().await;
        let worker = Worker::new(app.clone());
        tokio::spawn(WorkerRunner { worker, runner }.loop_forever());
    }
}
