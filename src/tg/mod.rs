//! Telegram client layer: session, userbot auth, peer caches and human-ish sending.
//! A port of kuni-master's `src/telegram/TelegramClientImpl.cpp` on top of Grammers.

pub mod human;

use std::collections::HashMap;
use std::sync::Arc;

use grammers_client::client::SignInError;
use grammers_client::tl as tl;
use grammers_client::message::{InputMessage, InputReactions, Message};
use grammers_client::peer::{Peer, User};
use grammers_client::Client;
use grammers_mtsender::SenderPool;
use grammers_session::storages::SqliteSession;
use grammers_session::types::{PeerId, PeerRef};
use grammers_session::Session as _;

use crate::tg::human::{err, SendGate, SendStreak, TYPING_DELAY_GRACE};

pub type TgResult<T> = Result<T, String>;

/// Everything that behaves like a person when sending messages on Telegram.
pub struct Tg {
    pub client: Client,
    pub session: Arc<SqliteSession>,
    peers: tokio::sync::Mutex<HashMap<i64, Peer>>,
    gate: SendGate,
    streaks: tokio::sync::Mutex<HashMap<i64, SendStreak>>,
    typing_min_wpm: std::sync::atomic::AtomicI32,
    typing_max_wpm: std::sync::atomic::AtomicI32,
}

impl Tg {
    /// Creates the sender pool and a client handle. The pool's runner must be spawned
    /// by the caller (`runner.run()` inside a tokio task).
    pub async fn setup(session_path: &str, api_id: i32) -> anyhow::Result<(Arc<Self>, SenderPool)> {
        let session = Arc::new(SqliteSession::open(session_path).await?);
        let pool = SenderPool::new(session.clone(), api_id);
        let client = Client::new(pool.handle.clone());
        let tg = Arc::new(Self {
            client,
            session,
            peers: tokio::sync::Mutex::new(HashMap::new()),
            gate: SendGate::new(20),
            streaks: tokio::sync::Mutex::new(HashMap::new()),
            typing_min_wpm: std::sync::atomic::AtomicI32::new(120),
            typing_max_wpm: std::sync::atomic::AtomicI32::new(150),
        });
        Ok((tg, pool))
    }

    /// Refreshes the typing simulation knobs from config.
    pub fn set_typing_speed(&self, min_wpm: i32, max_wpm: i32) {
        self.typing_min_wpm
            .store(min_wpm, std::sync::atomic::Ordering::Relaxed);
        self.typing_max_wpm
            .store(max_wpm, std::sync::atomic::Ordering::Relaxed);
    }

    /// Interactive sign in (phone -> code -> optional 2FA password) via stdin,
    /// kuni's `TelegramClientImpl` authorization flow.
    pub async fn authorize_interactive(&self, api_hash: &str) -> TgResult<()> {
        use std::io::Write;
        if self.client.is_authorized().await.map_err(err)? {
            println!("Already authorized.");
            return Ok(());
        }
        println!("Not authorized. Signing in to Telegram...");
        print!("Phone (with country code): ");
        std::io::stdout().flush().ok();
        let mut phone = String::new();
        std::io::stdin().read_line(&mut phone).map_err(err)?;
        let phone = phone.trim().to_string();

        let token = self
            .client
            .request_login_code(&phone, api_hash)
            .await
            .map_err(err)?;

        loop {
            print!("Code (sent to any official Telegram client): ");
            std::io::stdout().flush().ok();
            let mut code = String::new();
            std::io::stdin().read_line(&mut code).map_err(err)?;
            let code = code.trim().to_string();
            match self.client.sign_in(&token, &code).await {
                Ok(me) => {
                    println!("Signed in as {}.", me.full_name());
                    return Ok(());
                }
                Err(SignInError::InvalidCode) => {
                    println!("Invalid code, try again.");
                    continue;
                }
                Err(SignInError::PasswordRequired(password_token)) => {
                    let mut password_token = password_token;
                    loop {
                        print!("2FA password: ");
                        std::io::stdout().flush().ok();
                        let mut pwd = String::new();
                        std::io::stdin().read_line(&mut pwd).map_err(err)?;
                        match self
                            .client
                            .check_password(password_token, pwd.trim().as_bytes())
                            .await
                        {
                            Ok(me) => {
                                println!("Signed in as {}.", me.full_name());
                                return Ok(());
                            }
                            Err(SignInError::InvalidPassword(token)) => {
                                password_token = token;
                                println!("Invalid password, try again.");
                                continue;
                            }
                            Err(err2) => return Err(format!("2FA check failed: {err2}")),
                        }
                    }
                }
                Err(err2) => return Err(format!("sign in failed: {err2}")),
            }
        }
    }

    /// Telegram user id of the logged-in account (kuni's `myId()`).
    pub async fn my_id(&self) -> TgResult<i64> {
        let me = self.client.get_me().await.map_err(err)?;
        self.remember_peer(&Peer::User(me.clone())).await;
        me.id().bot_api_dialog_id().ok_or_else(|| "cannot get own id".to_string())
    }

    /// Resolves a Bot-API-style dialog id (what the LLM sees everywhere) into a `PeerRef`
    /// usable in grammers' calls, preferring the session cache's access hash.
    pub async fn peer_ref(&self, dialog_id: i64) -> TgResult<PeerRef> {
        let id = PeerId::from_bot_api_dialog_id(dialog_id)
            .ok_or_else(|| format!("invalid chat id {dialog_id}"))?;
        if let Ok(Some(peer_ref)) = self.session.peer_ref(id).await {
            return Ok(peer_ref);
        }
        if let Some(peer) = self.peers.lock().await.get(&dialog_id) {
            if let Ok(Some(peer_ref)) = peer.to_ref().await {
                return Ok(peer_ref);
            }
        }
        Ok(id.to_ambient_ref())
    }

    /// Caches a peer (chat or user) by its Bot API dialog id.
    pub async fn remember_peer(&self, peer: &Peer) {
        if let Some(id) = peer.id().bot_api_dialog_id() {
            self.peers.lock().await.insert(id, peer.clone());
        }
    }

    /// Cached peer info by Bot API dialog id, or None.
    pub async fn chat(&self, dialog_id: i64) -> Option<Peer> {
        self.peers.lock().await.get(&dialog_id).cloned()
    }

    /// Loads recent dialogs into the peer cache (the kuni `getChats(chatListMain, N)` warm-up).
    /// Returns (dialog_id, peer) list, oldest first like kuni (reversed getChats result).
    pub async fn warm_cache_dialogs(&self, limit: usize) -> TgResult<Vec<(i64, Peer)>> {
        let mut ids = Vec::new();
        let mut iter = self.client.iter_dialogs().limit(limit);
        while let Some(dialog) = iter.next().await.map_err(err)? {
            let peer = dialog.peer;
            if let Some(id) = peer.id().bot_api_dialog_id() {
                self.peers.lock().await.insert(id, peer.clone());
                ids.push((id, peer));
            }
        }
        ids.reverse();
        Ok(ids)
    }

    /// Fetches a window of history. Returns messages **oldest first**, trimming
    /// everything already present in the context (`until_msg_seen`),
    /// or capped by the accumulated character budget (kuni's pagination loop).
    pub async fn load_history(
        &self,
        dialog_id: i64,
        limit: usize,
        until_msg_seen: Option<i32>,
        max_chars: usize,
    ) -> TgResult<Vec<Message>> {
        let peer = self.peer_ref(dialog_id).await?;
        let mut out = Vec::new();
        let mut chars = 0;
        let mut iter = self.client.iter_messages(peer).limit(limit.max(1));
        while let Some(message) = iter.next().await.map_err(err)? {
            let id = message.id();
            chars += message.text().chars().count();
            out.push(message);
            if let Some(until) = until_msg_seen {
                if id <= until {
                    break;
                }
            }
            if chars >= max_chars {
                break;
            }
        }
        // iter_messages yields newest first; reverse for oldest-first ordering.
        out.reverse();
        Ok(out)
    }

    /// Marks the whole chat as read (kuni's `viewMessages`).
    pub async fn mark_read(&self, dialog_id: i64) -> TgResult<()> {
        let peer = self.peer_ref(dialog_id).await?;
        self.client.mark_as_read(peer).await.map_err(err)?;
        Ok(())
    }

    /// Shows Telegram's online state (kuni's `setOption("online", ...)`).
    pub async fn set_online(&self, online: bool) -> TgResult<()> {
        self.client
            .invoke(&tl::functions::account::UpdateStatus {
                offline: !online,
            })
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Sends a chat action one-shot ("typing...", "choosing a sticker...", etc).
    pub async fn send_action(
        &self,
        dialog_id: i64,
        action: tl::enums::SendMessageAction,
    ) -> TgResult<()> {
        let peer = self.peer_ref(dialog_id).await?;
        self.client.action(peer).oneshot(action).await.map_err(err)
    }

    /// The human-aware message sending pipeline (kuni's `telegramPostMessage`):
    /// rate gate -> typing simulation -> typo simulation -> send, with the
    /// `messagesInARow` guard (>= 10 in a row: fail).
    pub async fn send_text(
        &self,
        dialog_id: i64,
        text: &str,
        reply_to: Option<i32>,
    ) -> TgResult<()> {
        let texts = human::split_for_sending(text);
        if texts.is_empty() {
            return Err("empty message".to_string());
        }
        {
            let mut streaks = self.streaks.lock().await;
            let streak = streaks.entry(dialog_id).or_insert_with(SendStreak::new);
            human::streak_decay(streak);
            if streak.in_row >= 10 {
                return Err(
                    "too many messages in a row (>10); wait for people to reply first"
                        .to_string(),
                );
            }
        }
        let peer = self.peer_ref(dialog_id).await?;
        for part in texts {
            self.gate.wait().await;
            let mut rng = rand::rng();

            // Typing simulation: only when the previous message in the chat is recent
            // (kuni: typing delay applies if the previous message was sent < 5s ago).
            let recent = {
                let streaks = self.streaks.lock().await;
                streaks
                    .get(&dialog_id)
                    .map(|s| s.last.elapsed() < TYPING_DELAY_GRACE)
                    .unwrap_or(false)
            };
            if recent {
                let wpm = human::random_wpm(
                    self.typing_min_wpm.load(std::sync::atomic::Ordering::Relaxed),
                    self.typing_max_wpm.load(std::sync::atomic::Ordering::Relaxed),
                    &mut rng,
                );
                let delay = human::typing_delay(part.chars().count(), wpm);
                let _ = self
                    .send_action(
                        dialog_id,
                        tl::enums::SendMessageAction::SendMessageTypingAction,
                    )
                    .await;
                tokio::time::sleep(delay).await;
            }

            // Typo simulation per message.
            let part = human::apply_typo(&part, &mut rng);

            let input = InputMessage::new().text(part).reply_to(reply_to);
            self.client.send_message(peer, input).await.map_err(err)?;
            {
                let mut streaks = self.streaks.lock().await;
                let streak = streaks.entry(dialog_id).or_insert_with(SendStreak::new);
                streak.in_row += 1;
                streak.last = std::time::Instant::now();
            }
        }
        Ok(())
    }

    /// Edits a previously sent message's text.
    pub async fn edit_text(
        &self,
        dialog_id: i64,
        message_id: i32,
        text: &str,
    ) -> TgResult<()> {
        let peer = self.peer_ref(dialog_id).await?;
        self.client
            .edit_message(peer, message_id, InputMessage::new().text(text))
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Deletes messages (revoke=true, kuni's `deleteMessages(chat_id, ids, true)`).
    pub async fn delete_messages(
        &self,
        dialog_id: i64,
        message_ids: &[i32],
    ) -> TgResult<usize> {
        let peer = self.peer_ref(dialog_id).await?;
        self.client
            .delete_messages(peer, message_ids)
            .await
            .map_err(err)
    }

    /// Forwards messages between chats and optionally adds a follow-up comment.
    pub async fn forward_messages(
        &self,
        to_chat: i64,
        from_chat: i64,
        message_ids: &[i32],
        comment: Option<&str>,
    ) -> TgResult<()> {
        let to = self.peer_ref(to_chat).await?;
        let from = self.peer_ref(from_chat).await?;
        self.client
            .forward_messages(to, message_ids, from)
            .await
            .map_err(err)?;
        if let Some(comment) = comment {
            self.send_text(to_chat, comment, None).await?;
        }
        Ok(())
    }

    /// Adds an emoji reaction to a message.
    pub async fn react(&self, dialog_id: i64, message_id: i32, emoji: &str) -> TgResult<()> {
        let peer = self.peer_ref(dialog_id).await?;
        let reactions = InputReactions::emoticon(emoji);
        self.client
            .send_reactions(peer, message_id, reactions)
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Resolves '@username' text into a peer.
    pub async fn resolve_username(&self, username: &str) -> TgResult<Option<Peer>> {
        let peer = self
            .client
            .resolve_username(username.trim_start_matches('@'))
            .await
            .map_err(err)?;
        if let Some(peer) = &peer {
            self.remember_peer(peer).await;
        }
        Ok(peer)
    }

    /// List of chat admins — the flood-specific `get_chat_admins` backing tool.
    pub async fn chat_admins(&self, dialog_id: i64) -> TgResult<Vec<(User, Option<String>)>> {
        let peer = self.peer_ref(dialog_id).await?;
        let mut out = Vec::new();
        let mut iter = self
            .client
            .iter_participants(peer)
            .filter(tl::enums::ChannelParticipantsFilter::ChannelParticipantsAdmins);
        while let Some(participant) = iter.next().await.map_err(err)? {
            let rank = match &participant.role {
                grammers_client::peer::Role::Admin(admin) => {
                    admin.rank().map(str::to_string)
                }
                _ => None,
            };
            let user = participant.user;
            self.remember_peer(&Peer::User(user.clone())).await;
            out.push((user, rank));
        }
        Ok(out)
    }
}
