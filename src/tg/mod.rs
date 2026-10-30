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

/// One recent dialog summary for the chat list tools.
pub struct ChatListEntry {
    pub id: i64,
    pub peer: Peer,
    pub unread: u32,
    pub last_message: Option<Message>,
}

/// Everything that behaves like a person when sending messages on Telegram.
pub struct Tg {
    pub client: Client,
    pub session: Arc<SqliteSession>,
    peers: tokio::sync::Mutex<HashMap<i64, Peer>>,
    gate: SendGate,
    streaks: tokio::sync::Mutex<HashMap<i64, SendStreak>>,
    typing_min_wpm: std::sync::atomic::AtomicI32,
    typing_max_wpm: std::sync::atomic::AtomicI32,
    /// kuni's global embedding cache for anti-repeat (text -> embedding).
    pub anti_repeat_cache: std::sync::Mutex<HashMap<String, Vec<f64>>>,
    /// kuni's shared "head start" relaxation value, bumped on repeats and cleared on
    /// success (kuni's `pEventLoop global soak`).
    pub anti_repeat_head_start: std::sync::Mutex<f64>,
    /// Cached contact list ids for the CONTACTS_ONLY lockdown mode (kuni's static
    /// contacts lookup).
    contacts: tokio::sync::Mutex<Option<Vec<i64>>>,
    /// Cached own user id (0 = unset), kuni's `myId()`.
    my_id_cache: std::sync::atomic::AtomicI64,
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
            anti_repeat_cache: std::sync::Mutex::new(HashMap::new()),
            anti_repeat_head_start: std::sync::Mutex::new(0.0),
            contacts: tokio::sync::Mutex::new(None),
            my_id_cache: std::sync::atomic::AtomicI64::new(0),
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
        let cached = self.my_id_cache.load(std::sync::atomic::Ordering::SeqCst);
        if cached != 0 {
            return Ok(cached);
        }
        let me = self.client.get_me().await.map_err(err)?;
        self.remember_peer(&Peer::User(me.clone())).await;
        let id = me
            .id()
            .bot_api_dialog_id()
            .ok_or_else(|| "cannot get own id".to_string())?;
        self.my_id_cache.store(id, std::sync::atomic::Ordering::SeqCst);
        Ok(id)
    }

    /// Cached-only variant of `my_id` for hot paths (returns the id without cache
    /// refill; callers use `my_id()` once at startup).
    pub async fn my_id_cached(&self) -> TgResult<i64> {
        self.my_id().await
    }

    /// Best-effort chat title from the peer cache (fallback "<unknown>").
    pub async fn chat_title(&self, dialog_id: i64) -> String {
        let peer = match self.chat(dialog_id).await {
            Some(peer) => peer,
            None => {
                let resolved = match self.peer_ref(dialog_id).await.ok() {
                    Some(peer_ref) => self.client.resolve_peer(peer_ref).await.ok(),
                    None => None,
                };
                match resolved {
                    Some(peer) => {
                        self.remember_peer(&peer).await;
                        peer
                    }
                    None => return "<unknown>".to_string(),
                }
            }
        };
        crate::llmui::dialog::peer_title(&peer)
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

    /// Fetches recent dialogs into the peer cache and returns their summaries
    /// (kuni's `getChats(chatListMain, N)`), oldest first like kuni.
    pub async fn get_chats(&self, limit: usize) -> TgResult<Vec<ChatListEntry>> {
        let mut out = Vec::new();
        let mut iter = self.client.iter_dialogs().limit(limit);
        while let Some(dialog) = iter.next().await.map_err(err)? {
            let peer = dialog.peer;
            if let Some(id) = peer.id().bot_api_dialog_id() {
                let unread = crate::llmui::dialog::dialog_unread(&dialog.raw);
                out.push(ChatListEntry {
                    id,
                    peer,
                    unread,
                    last_message: dialog.last_message,
                });
            }
        }
        out.reverse();
        Ok(out)
    }

    /// Startup warm cache (legacy helper, kept for `App::init`).
    pub async fn warm_cache_dialogs(&self, limit: usize) -> TgResult<Vec<ChatListEntry>> {
        self.get_chats(limit).await
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
    /// Returns the message ids sent (one per split part).
    pub async fn send_text(
        &self,
        dialog_id: i64,
        text: &str,
        reply_to: Option<i32>,
    ) -> TgResult<Vec<i32>> {
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
        let mut sent_ids = Vec::new();
        for part in texts {
            self.gate.wait().await;
            // StdRng (not ThreadRng, which is !Send) so it may live across awaits.
            let mut rng =
                rand::rngs::StdRng::from_rng(&mut rand::rng()).expect("rng seed");

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

            // LLM uses "—" (long dash) instead of "-". Fix it (kuni's sendTelegramMessage).
            let part = part.replace('—', "-");

            let input = InputMessage::new().text(part).reply_to(reply_to);
            let sent = self.client.send_message(peer, input).await.map_err(err)?;
            sent_ids.push(sent.id());
            {
                let mut streaks = self.streaks.lock().await;
                let streak = streaks.entry(dialog_id).or_insert_with(SendStreak::new);
                streak.in_row += 1;
                streak.last = std::time::Instant::now();
            }
        }
        Ok(sent_ids)
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

    /// Prevents a user from DM-ing the account (kuni's `setMessageSenderBlockList`,
    /// TL contacts.block).
    pub async fn block_user(&self, user_dialog_id: i64) -> TgResult<()> {
        let peer: tl::enums::InputPeer = self.peer_ref(user_dialog_id).await?.into();
        self.client
            .invoke(&tl::functions::contacts::Block {
                id: peer,
                my_stories_from: false,
            })
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Deletes the whole chat history with `revoke=true` (kuni's `deleteChatHistory`).
    pub async fn delete_chat_history(&self, dialog_id: i64) -> TgResult<()> {
        let peer: tl::enums::InputPeer = self.peer_ref(dialog_id).await?.into();
        self.client
            .invoke(&tl::functions::messages::DeleteHistory {
                peer,
                max_id: 0,
                min_date: None,
                max_date: None,
                just_clear: false,
                revoke: true,
            })
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Leaves a channel/supergroup chat (kuni's `leaveChat`).
    /// Basic groups are not supported by Grammers' layer here — an error is reported.
    pub async fn leave_chat(&self, dialog_id: i64) -> TgResult<()> {
        let channel = match self.chat(dialog_id).await {
            Some(Peer::Channel(channel)) => channel.raw,
            _ => {
                return Err(
                    "cannot leave this chat (only channels/supergroups are supported)"
                        .to_string(),
                )
            }
        };
        let input_channel = tl::types::InputChannel {
            channel_id: channel.id,
            access_hash: channel.access_hash.unwrap_or(0),
        };
        self.client
            .invoke(&tl::functions::channels::LeaveChannel {
                channel: tl::enums::InputChannel::Channel(input_channel),
            })
            .await
            .map_err(err)?;
        Ok(())
    }

    /// Resolves a "@username" string into the Bot-API dialog id of that chat.
    pub async fn resolve_chat_id(&self, query: &str) -> TgResult<i64> {
        let peer = self.resolve_username(query).await?;
        peer.and_then(|p| p.id().bot_api_dialog_id())
            .ok_or_else(|| format!("could not resolve \"{query}\" to a chat"))
    }

    /// Joins a channel/supergroup chat (kuni's `joinChat`).
    pub async fn join_chat(&self, dialog_id: i64) -> TgResult<()> {
        let peer_ref = self.peer_ref(dialog_id).await?;
        self.client.join_chat(peer_ref).await.map_err(err)?;
        Ok(())
    }

    /// Joins a chat by its invite link and returns the dialog id (kuni's
    /// `joinChatByInviteLink`).
    pub async fn join_chat_by_invite_link(&self, invite_link: &str) -> TgResult<i64> {
        let peer = self
            .client
            .accept_invite_link(invite_link)
            .await
            .map_err(err)?
            .ok_or_else(|| "could not join chat: no chat in the invite result".to_string())?;
        self.remember_peer(&peer).await;
        let id = peer
            .id()
            .bot_api_dialog_id()
            .ok_or_else(|| "could not determine joined chat id".to_string())?;
        Ok(id)
    }

    /// The account's contact list (TL contacts.getContacts), cached once like kuni's
    /// static contacts lookup.
    pub async fn get_contact_ids(&self) -> TgResult<Vec<i64>> {
        if let Some(contacts) = self.contacts.lock().await.as_ref() {
            return Ok(contacts.clone());
        }
        let result: tl::enums::contacts::Contacts = self
            .client
            .invoke(&tl::functions::contacts::GetContacts { hash: 0 })
            .await
            .map_err(err)?;
        let mut ids = Vec::new();
        if let tl::enums::contacts::Contacts::Contacts(contacts) = result {
            for user in contacts.users {
                let peer =
                    grammers_client::peer::Peer::from_user(&self.client, user);
                self.remember_peer(&peer).await;
                if let Some(id) = peer.id().bot_api_dialog_id() {
                    ids.push(id);
                }
            }
        }
        *self.contacts.lock().await = Some(ids.clone());
        Ok(ids)
    }
}
