//! Application core: config application, warm-up, notification building, chat
//! opening and lockdown checks.
//! Port of kuni's `main.cpp` `App` class + `AppBase::getSystemPrompt` (Phase 2).
//!
//! Event-flow functions are free-standing and take `Arc<App>` explicitly, because
//! notification/tool handlers capture an owned `Arc<App>`.

use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex as StdMutex};

use grammers_client::peer::Peer;
use grammers_client::update::Update;
use tokio::sync::{Mutex as AsyncMutex, RwLock};

use crate::config::{ConfigHub, LockdownMode};
use crate::diary::Diary;
use crate::llm::Llm;
use crate::llmui;
use crate::notifications::{Notification, NotificationManager};
use crate::prompts::{PromptStore, PromptVars};
use crate::tg::Tg;
use crate::tools::{self, Ctx};

/// kuni's `AppBase::WORKING_MEMORY_PATH` (relative to the working dir).
const WORKING_MEMORY_PATH: &str = "working_memory.md";
/// kuni's working dir: everything persistent lives here (`main.cpp:769` diary,
/// `AppBase` working memory).
pub const WORK_DIR: &str = "data";

/// The kind of a chat: determines the available tools and instructions
/// (kuni's chatType switch in `llmuiOpenTelegramChat`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChatKind {
    Private,
    Group,
    Channel,
}

/// Per-chat state persisting between sends (kuni's `State` in sendTelegramMessage).
pub struct OpenedChatState {
    pub msgs_in_a_row: AtomicU32,
    pub last_reply_to: std::sync::Mutex<i64>,
}

/// An open chat the worker converses in. Captured by the chat tool handlers.
pub struct OpenedChat {
    pub chat_id: i64,
    pub title: String,
    pub kind: ChatKind,
    pub is_admin: bool,
    /// Messages loaded into the context at open time, oldest first
    /// (kuni passes the loaded `array<message>` into send_telegram_message).
    pub history: Vec<grammers_client::message::Message>,
    /// The logged-in account's user id, for own-message detection.
    pub my_id: i64,
    pub state: OpenedChatState,
}

impl OpenedChat {
    pub fn history(&self) -> &[grammers_client::message::Message] {
        &self.history
    }
}

pub struct App {
    pub config: Arc<ConfigHub>,
    pub tg: Arc<Tg>,
    pub handle: grammers_mtsender::SenderPoolFatHandle,
    pub llm: Llm,
    pub prompts: Arc<PromptStore>,
    pub notifications: Arc<NotificationManager>,
    /// Dialog id of the chat the LLM currently generates a reply in
    /// (kuni's `mCurrentlyOpenedChat`, used by the typing indicator); 0 = none.
    pub currently_opened_chat: std::sync::atomic::AtomicI64,
    /// The long-term memory (kuni's `mDiary`).
    pub diary: Arc<Diary>,
    /// kuni's `mWorkingMemoryLock`: serializes diary dump flows.
    working_memory_lock: Arc<AsyncMutex<()>>,
    /// kuni's `mSystemPromptSuffix` (`onCleanContext` output), reset by dumps;
    /// `None` == recompute needed.
    system_prompt_suffix: RwLock<Option<String>>,
    /// Registered worker sessions for the shutdown dump (kuni's `mWorkers`).
    pub worker_sessions: StdMutex<Vec<Arc<StdMutex<Vec<crate::llm::Message>>>>>,
}

impl App {
    pub fn new(
        config: Arc<ConfigHub>,
        tg: Arc<Tg>,
        handle: grammers_mtsender::SenderPoolFatHandle,
        llm: Llm,
        prompts: Arc<PromptStore>,
        notifications: Arc<NotificationManager>,
        diary: Arc<Diary>,
    ) -> Self {
        Self {
            config,
            tg,
            handle,
            llm,
            prompts,
            notifications,
            diary,
            currently_opened_chat: std::sync::atomic::AtomicI64::new(0),
            working_memory_lock: Arc::new(AsyncMutex::new(())),
            system_prompt_suffix: RwLock::new(None),
            worker_sessions: StdMutex::new(Vec::new()),
        }
    }

    pub fn working_memory_path(&self) -> PathBuf {
        std::path::Path::new(WORK_DIR).join(WORKING_MEMORY_PATH)
    }

    /// The prompt variable substitutions for this configuration
    /// (kuni's `Prompts::reg` variables).
    pub async fn prompt_vars(&self) -> PromptVars {
        let config = self.config.get().await;
        PromptVars {
            character_name: config.general.character_name.clone(),
            character_nickname: config.general.character_nickname.clone(),
            papik_name: config.general.papik_name.clone(),
        }
    }

    /// kuni's `AppBase::getSystemPrompt`: system + character base + appearance + the
    /// cached working-memory suffix + the lockdown note.
    pub async fn get_system_prompt(&self) -> String {
        let config = self.config.get().await;
        let prompts = self.prompts.get(&self.prompt_vars().await).await;
        let mut suffix = self.system_prompt_suffix.write().await;
        if suffix.is_none() {
            *suffix = Some(self.on_clean_context().await);
        }
        let mut prompt = format!(
            "{}\n\n{}\n\n<your_appearance>\n{}\n</your_appearance>\n{}",
            prompts.system,
            prompts.character_base,
            prompts.character_appearance,
            suffix.clone().unwrap_or_default()
        );
        drop(suffix);
        if config.general.lockdown != LockdownMode::None {
            prompt +=
                "\nYou are currently under lockdown mode. This means you can access to subset of chats only.";
        }
        prompt
    }

    /// kuni's `AppBase::onCleanContext`: the `<things_to_remember>` suffix injected
    /// into the system prompt until the next diary dump.
    async fn on_clean_context(&self) -> String {
        let Ok(working_memory) = tokio::fs::read_to_string(self.working_memory_path()).await
        else {
            return String::new();
        };
        format!(
            r#"<things_to_remember>
{}
</things_to_remember>
<instructions>
Your behaviour must be highly influenced by "physical state" and "emotional state" mentioned above.

<example>
Emotional state: anger
...
send_telegram_message("text":"иди нахуй заебал")
</example>
<example>
Emotional state: amused
...
send_telegram_message("text":"мррр~")
</example>
</instruction>
"#,
            working_memory
        )
    }

    // ===================================================================
    // Diary (phase 3): context dump (AppBase::diaryDumpMessages).
    // ===================================================================

    /// kuni's `AppBase::diaryDumpMessages`: distills the session into diary entries
    /// + the working memory file, then clears the context and the prompt suffix.
    /// Diar reload runs on every path (kuni's `AUI_DEFER`).
    pub async fn diary_dump_messages(
        app: &Arc<App>,
        session: &Arc<StdMutex<Vec<crate::llm::Message>>>,
    ) -> Result<(), String> {
        let result = diary_dump_impl(app, session).await;
        app.diary.reload().await;
        result
    }

    async fn diary_summary(
        app: &Arc<App>,
        mut context: Vec<crate::llm::Message>,
    ) -> Result<Vec<crate::diary::EntryEx>, String> {
        let prompts = app.prompts.get(&app.prompt_vars().await).await;
        let system_prompt = app.get_system_prompt().await;
        context.push(crate::llm::Message::user(prompts.diary_save));
        loop {
            let (message, _) = app
                .llm
                .chat_streaming(Some(&system_prompt), &context, None, None)
                .await
                .map_err(|e| format!("diary summary LLM call failed: {e}"))?;
            if message.content.is_empty() {
                context.push(message);
                context.push(crate::llm::Message::user(
                    "Tools are not available; write the summary without tools.",
                ));
                continue;
            }
            return app.diary.diary_save_entries(&message.content).await;
        }
    }

    /// kuni's `util::importantThingsToRemember` chat pump.
    async fn important_things_to_remember(
        app: &Arc<App>,
        context: &[crate::llm::Message],
        previous_working_memory: &str,
    ) -> Result<String, String> {
        let name = app
            .config
            .get()
            .await
            .general
            .character_name
            .clone();
        let prompt = crate::diary::important_things_to_remember_prompt(&name, previous_working_memory);
        let mut messages = context.to_vec();
        messages.push(crate::llm::Message::user(prompt));

        let mut shit_check_triggered = false;
        loop {
            let system_prompt = app.get_system_prompt().await;
            let (message, _) = app
                .llm
                .chat_streaming(Some(&system_prompt), &messages, None, None)
                .await
                .map_err(|e| format!("working memory LLM call failed: {e}"))?;
            let mut content = message.content;
            if content.contains("tool_calls") || content.contains("ask") {
                // deepseek bug - attempts to use DSML to make a tool call.
                if !shit_check_triggered {
                    shit_check_triggered = true;
                    continue;
                }
            }
            // sometimes llm wraps with xml tags by itself, we don't need that
            content = content
                .replace("<things_to_remember>", "")
                .replace("</things_to_remember>", "");
            let content = content.trim_start_matches(|c: char| c != '\n' && c.is_whitespace());
            if content.is_empty() {
                // sometimes llm returns blank response - try again
                continue;
            }
            return Ok(content.to_string());
        }
    }

    /// Applies dynamic (hot-reloaded) config knobs to live structures.
    pub async fn refresh_config(&self) {
        let cfg = self.config.get().await;
        self.tg.set_typing_speed(
            cfg.misc.typing_simulation_min_wpm,
            cfg.misc.typing_simulation_max_wpm,
        );
    }

    // ------------------------------------------------------------------
    // lockdown (util/is_accessible_from_lockdown.cpp)
    // ------------------------------------------------------------------

    /// General lockdown for tools/chat opening (`general.lockdown`).
    pub async fn is_accessible_lockdown(&self, chat_id: i64) -> bool {
        lockdown_check(
            self.tg.clone(),
            self.config.get().await.general.lockdown,
            self.config.get().await.general.papik_chat_id,
            chat_id,
        )
        .await
    }

    /// Notification filter lockdown (`misc.chat_notification_filter`).
    pub async fn passes_notification_filter(&self, chat_id: i64) -> bool {
        let config = self.config.get().await;
        lockdown_check(
            self.tg.clone(),
            config.misc.chat_notification_filter,
            config.general.papik_chat_id,
            chat_id,
        )
        .await
    }

    /// What kind of a chat is behind this dialog id; `None` when unknown.
    pub async fn chat_kind(&self, chat_id: i64) -> Option<ChatKind> {
        let peer = match self.tg.chat(chat_id).await {
            Some(peer) => peer,
            None => {
                let peer_ref = self.tg.peer_ref(chat_id).await.ok()?;
                let peer = self.tg.client.resolve_peer(peer_ref).await.ok()?;
                self.tg.remember_peer(&peer).await;
                peer
            }
        };
        match peer {
            Peer::User(_) => Some(ChatKind::Private),
            Peer::Group(_) => Some(ChatKind::Group),
            Peer::Channel(c) => {
                if c.raw.broadcast {
                    Some(ChatKind::Channel)
                } else {
                    Some(ChatKind::Group)
                }
            }
            _ => None,
        }
    }

    /// Membership rule: kuni checks `getChatMember`; we treat chats present in the
    /// dialog cache as member chats (the dialog list contains only member chats).
    pub async fn is_member_of(&self, chat_id: i64) -> bool {
        self.tg.chat(chat_id).await.is_some()
    }

    /// Am I an administrator of this group chat? (kuni's
    /// chatMemberStatusAdministrator/Creator detection)
    pub async fn is_group_admin(&self, chat_id: i64) -> bool {
        if matches!(self.chat_kind(chat_id).await, None | Some(ChatKind::Private)) {
            return false;
        }
        let Ok(admins) = self.tg.chat_admins(chat_id).await else {
            return false;
        };
        let Ok(my_id) = self.tg.my_id().await else {
            return false;
        };
        admins
            .iter()
            .any(|(u, _)| u.id().bot_api_dialog_id() == Some(my_id))
    }
}

/// kuni's `util::isAccessibleFromLockdown(telegram, chatId, lockdown)`.
async fn lockdown_check(tg: Arc<Tg>, mode: LockdownMode, papik_chat_id: i64, chat_id: i64) -> bool {
    match mode {
        LockdownMode::None => true,
        LockdownMode::ContactsOnly => {
            if chat_id < 0 {
                // a group chat that kuni participates in, ok.
                return true;
            }
            let contacts = tg.get_contact_ids().await.unwrap_or_default();
            if contacts.contains(&chat_id) {
                return true;
            }
            // fallthrough
            chat_id == papik_chat_id
        }
        LockdownMode::PapikOnly => chat_id == papik_chat_id,
    }
}

// ===========================================================================
// Startup warm-up (`sendNotificationsOnInit`, main.cpp:234)
// ===========================================================================

pub async fn init(app: &Arc<App>) -> anyhow::Result<()> {
    app.tg.my_id().await.map_err(anyhow::Error::msg)?;
    let cfg = app.config.get().await;
    if !cfg.misc.check_chats_on_startup {
        return Ok(());
    }
    let chats = app.tg.get_chats(200).await.map_err(anyhow::Error::msg)?;
    println!("Cached {} recent dialogs.", chats.len());
    // tdlib does not send notifications for unread chats on program startup.
    // We fix this by pushing the last message of every unread chat through the
    // same handler as live events.
    for entry in chats.iter().filter(|c| c.unread > 0) {
        let Some(message) = &entry.last_message else {
            continue;
        };
        handle_new_message(app, message).await;
    }
    Ok(())
}

// ===========================================================================
// Telegram event handling (main.cpp `handleTelegramEvent`)
// ===========================================================================

pub async fn on_update(app: &Arc<App>, update: Update) {
    if let Update::NewMessage(message) = &update {
        handle_new_message(app, message).await;
    }
}

async fn handle_new_message(app: &Arc<App>, message: &grammers_client::message::Message) {
    if message.outgoing() {
        return;
    }
    let Some(chat_id) = message.peer().and_then(|p| p.id().bot_api_dialog_id()) else {
        return;
    };
    let Some(sender_id) = message.sender_id().and_then(|id| id.bot_api_dialog_id()) else {
        return;
    };
    let Ok(my_id) = app.tg.my_id().await else {
        return;
    };
    if sender_id == my_id {
        return;
    }

    // Check lockdown mode - only allow PAPIK_CHAT_ID if lockdown is enabled
    if !app.is_accessible_lockdown(chat_id).await {
        return;
    }
    if !app.passes_notification_filter(chat_id).await {
        return;
    }

    // TODO(phase 4): muted chats get an 80% ignore probability
    // (needs notify settings from the dialog cache).

    let title = app.tg.chat_title(chat_id).await;
    let cfg = app.config.get().await;

    let mut notification = format!("<notification chat_id=\"{chat_id}\">\n");
    if sender_id == chat_id {
        // /version command (kuni's tryHandleCmd, private chats only).
        if message.text() == "/version" {
            let version = format!("kuni-rs {}\nKernel: kuni", env!("CARGO_PKG_VERSION"));
            let _ = app.tg.send_text(chat_id, &version, None).await;
            return;
        }
        notification += &format!(
            "You received a direct message from {title} (chat_id = {chat_id})"
        );
    } else if sender_id != 0 {
        let sender_name = app.tg.chat_title(sender_id).await;
        notification += &format!(
            "{sender_name} (user_id = {sender_id}) sent a message in group chat \"{title}\" (chat_id = {chat_id})"
        );
    } else {
        notification +=
            &format!("Channel \"{title}\" (chat_id={chat_id}) created a new post\n");
    }
    notification += "\n</notification>\n";
    notification += "You don't have any chat open. Use #open tool to open the chat";

    // The priority of the notification (papik always kicks in first;
    // TODO(phase 4): pinned chats get priority 100 when wake_up_on_pinned_chat).
    let priority = if sender_id == cfg.general.papik_chat_id {
        1000
    } else {
        0
    };

    let actions = tools::build_open_notification_actions(app, chat_id, &title);
    app.notifications
        .pass_notification(Notification {
            message: notification,
            actions,
            priority,
            pin: Some(format!("<chat id=\"{chat_id}\" />")),
        })
        .await;
}

// ===========================================================================
// Chat opening (kuni's `llmuiOpenTelegramChat`)
// ===========================================================================

/// Hard cap of messages loaded in one open (the normal limit is the character
/// budget below).
const HISTORY_HARD_LIMIT: usize = 500;

/// Opens the chat for the worker: marks the open chat (kuni's
/// `mCurrentlyOpenedChat`), removes pending notifications of this chat, loads the
/// history delta (using `ctx.seen` for continuation, kuni's temporaryContext
/// scan), formats it, installs the chat tool set replacing `ctx.tools` and returns
/// the natural-language result.
pub async fn open_telegram_chat(app: &Arc<App>, ctx: &Ctx, chat_id: i64) -> Result<String, String> {
    // Check lockdown mode - only allow PAPIK_CHAT_ID if lockdown is enabled
    if !app.is_accessible_lockdown(chat_id).await {
        tracing::error!(
            "Error: Lockdown mode is enabled. You can only open chat with ID {} (PAPIK_CHAT_ID).",
            app.config.get().await.general.papik_chat_id
        );
        return Ok("No such chat".to_string());
    }

    let cfg = app.config.get().await;

    app.tg.set_online(true).await.map_err(err_display)?;
    app.notifications
        .remove_notifications(&format!("<notification chat_id=\"{chat_id}\">"))
        .await;

    let Some(kind) = app.chat_kind(chat_id).await else {
        return Ok("No such chat".to_string());
    };

    let title = app.tg.chat_title(chat_id).await;
    let my_id = app.tg.my_id().await.map_err(err_display)?;
    let is_admin = kind != ChatKind::Private && app.is_group_admin(chat_id).await;
    let is_member = kind == ChatKind::Private || app.is_member_of(chat_id).await;

    // ---- history loading (kuni's getChatHistory paging) -----------------------
    // newest -> older; stop at `chat_max_history_length` characters of content, or
    // right after the messages the worker context already contains (delta mode).
    let peer_ref = app.tg.peer_ref(chat_id).await.map_err(err_display)?;
    let budget = cfg.misc.chat_max_history_length.min(200_000);
    let seen_top = ctx.seen.lock().unwrap().get(&chat_id).copied();

    let mut history: Vec<grammers_client::message::Message> = Vec::new();
    {
        // Grammers yields messages newest-first; with a `seen` marker we continue
        // right after it in ascending order (delta reshape, oldest first).
        let mut iter = match seen_top {
            Some(top) => app
                .tg
                .client
                .iter_messages(peer_ref)
                .offset_id(top)
                .reverse(true),
            None => app
                .tg
                .client
                .iter_messages(peer_ref)
                .limit(HISTORY_HARD_LIMIT),
        };
        let mut length = 0usize;
        while let Some(m) = iter.next().await.map_err(err_display)? {
            length += m.text().chars().count().saturating_add(40);
            history.push(m);
            if length >= budget {
                break;
            }
        }
        if seen_top.is_none() {
            // newest-first load; kuni formats oldest-first (`range | reverse`).
            history.reverse();
        }
    }
    tracing::info!("Loaded {} message(s): {title}", history.len());

    let mut result = String::new();
    if history.is_empty() && cfg.misc.can_write_to_a_new_person {
        result += "This chat is empty! Only proceed if you looked up a @username and it led you here.\n";
        result +=
            "Only write what you have to say to the chat; if someone asked you to text this person, just text them.\n";
        result +=
            "If you try to get back to the original chat and type something, you will be sending an extra message to the wrong chat.";
    }
    for m in &history {
        result += &llmui::format_history_message(
            &app.tg,
            m,
            my_id,
            &cfg.general.character_name,
            0,
            "message",
        )
        .await?;
        result.push('\n');
    }
    if !history.is_empty() {
        // read_all=true (kuni's viewMessages with force_read)
        let _ = app.tg.mark_read(chat_id).await;
        result = format!(
            "You switched to the chat \"{title}\" in Telegram. You see last messages:\n{result}"
        );
    } else {
        result = format!("You switched to the chat \"{title}\" in Telegram.\n{result}");
    }

    let opened = Arc::new(OpenedChat {
        chat_id,
        title: title.clone(),
        kind,
        is_admin,
        history,
        my_id,
        state: OpenedChatState {
            msgs_in_a_row: AtomicU32::new(0),
            last_reply_to: std::sync::Mutex::new(0),
        },
    });

    // ---- instructions & tool set replacement (kuni's per-kind blocks) ---------

    // Mirror the official Telegram client behavior: if you open a group chat/channel
    // you haven't joined, you only see a "Join" button (plus the ability to react to
    // messages) instead of a text field.
    if kind != ChatKind::Private && !is_member {
        result += &format!(
            "\n<instructions>\nYou are NOT a member of \"{title}\". You can't send messages here; you can only #react_with_emoji to messages, or use\n#join_chat to become a member (after which you'll be able to fully participate the next time you open this chat).\n</instructions>"
        );
        ctx.replace_tools(tools::build_chat_tools(app.clone(), opened, &cfg, false));
        app.currently_opened_chat
            .store(chat_id, std::sync::atomic::Ordering::SeqCst);
        return Ok(result);
    }

    match kind {
        ChatKind::Private => {
            let prompts = app.prompts.get(&app.prompt_vars().await).await;
            result += &format!(
                "\n<instructions>\nYou are in private chat with {title} (also known as direct messages or DM).\n\n{}\n</instructions>",
                prompts.messages_epilogue
            );
        }
        ChatKind::Group => {
            let prompts = app.prompts.get(&app.prompt_vars().await).await;
            result += &format!(
                "\n<instructions>\nYou are in group chat called \\\"{title}\\\".\n\n{}\n</instructions>",
                prompts.messages_epilogue
            );
        }
        ChatKind::Channel => {
            result += &format!(
                "\n<instructions>\nYou are in telegram channel (also known as supergroup) called \\\"{title}\\\".\nPay close attention to these messages. Acquire context from them. You can't respond in telegram channels\n(#send_telegram_message tool is not available). Instead, do what you usually do when reading newsletters: reflect and reason\non them.\nSome channels have reactions enabled. In that case, you can sometimes react with #react_with_emoji to express your feelings about a message, but you can't send a full reply.\n\nForwarding posts:\nIf you find a post genuinely interesting, funny, or relevant to someone you know — you can forward it to another chat\nusing #forward_message. Be selective: only forward posts that are truly worth sharing. You can add a short comment\nexpressing your reaction. Use #get_telegram_chats to find the destination chat_id if needed.\nDo NOT forward ads, sponsored posts, or low-value content.\n</instructions>"
            );
            // no send_telegram_message for channels
            ctx.replace_tools(tools::build_chat_tools(app.clone(), opened, &cfg, is_member));
            app.currently_opened_chat
                .store(chat_id, std::sync::atomic::Ordering::SeqCst);
            return Ok(result);
        }
    }

    ctx.replace_tools(tools::build_chat_tools(app.clone(), opened, &cfg, is_member));
    app.currently_opened_chat
        .store(chat_id, std::sync::atomic::Ordering::SeqCst);
    Ok(result)
}

fn err_display(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The body of `diary_dump_messages` (the `AUI_DEFER` reload is handled by the
/// caller).
async fn diary_dump_impl(
    app: &Arc<App>,
    session: &Arc<StdMutex<Vec<crate::llm::Message>>>,
) -> Result<(), String> {
    // kuni's try_lock loop over mWorkingMemoryLock.
    let lock = app.working_memory_lock.clone();
    let _guard = loop {
        if let Some(guard) = lock.try_lock() {
            break guard;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    };

    if session.lock().unwrap().is_empty() {
        return Ok(());
    }

    let previous_working_memory = tokio::fs::read_to_string(app.working_memory_path())
        .await
        .unwrap_or_default();

    // kuni launches the "important things" future before the diary summary so the
    // two LLM chats run concurrently; only the result is awaited afterwards.
    let important = {
        let app2 = app.clone();
        let context = session.lock().unwrap().clone();
        tokio::spawn(async move {
            App::important_things_to_remember(&app2, &context, &previous_working_memory).await
        })
    };

    let context = session.lock().unwrap().clone();
    App::diary_summary(app, context).await?;

    let working_memory_md = important
        .await
        .map_err(|e| format!("working memory LLM task failed: {e}"))??;
    tokio::fs::write(app.working_memory_path(), working_memory_md)
        .await
        .map_err(|e| format!("failed to write {}: {e}", app.working_memory_path().display()))?;

    session.lock().unwrap().clear();
    *app.system_prompt_suffix.write().await = None;
    Ok(())
}
