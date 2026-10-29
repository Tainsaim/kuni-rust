//! Application core: config application, warm-up and update handling.
//! Port of the relevant parts of kuni's `main.cpp` / `AppBase`
//! (Phase 1 slice: no LLM/notifications yet).

use std::sync::Arc;

use grammers_client::update::Update;
use grammers_mtsender::SenderPoolFatHandle;

use crate::config::ConfigHub;
use crate::llmui;
use crate::tg::{Tg, TgResult};

pub struct App {
    pub config: Arc<ConfigHub>,
    pub tg: Arc<Tg>,
    pub handle: SenderPoolFatHandle,
    my_id: tokio::sync::Mutex<Option<i64>>,
}

impl App {
    pub async fn new(
        config: Arc<ConfigHub>,
        tg: Arc<Tg>,
        handle: SenderPoolFatHandle,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            config,
            tg,
            handle,
            my_id: tokio::sync::Mutex::new(None),
        })
    }

    /// Telegram user id of the logged-in account — cached, kuni's `myId()`.
    pub async fn my_id(&self) -> TgResult<i64> {
        let mut guard = self.my_id.lock().await;
        if let Some(id) = *guard {
            return Ok(id);
        }
        let id = self.tg.my_id().await?;
        *guard = Some(id);
        Ok(id)
    }

    /// Applies dynamic (hot-reloaded) config knobs to live structures.
    pub async fn refresh_config(&self) {
        let cfg = self.config.get().await;
        self.tg.set_typing_speed(
            cfg.misc.typing_simulation_min_wpm,
            cfg.misc.typing_simulation_max_wpm,
        );
    }

    /// Startup warm-up (`sendNotificationsOnInit` in kuni): loads recent dialogs into
    /// the peer cache so later lookups by dialog id resolve access hashes.
    pub async fn init(&self) -> anyhow::Result<()> {
        let cfg = self.config.get().await;
        if !cfg.misc.check_chats_on_startup {
            return Ok(());
        }
        let chats = self
            .tg
            .warm_cache_dialogs(200)
            .await
            .map_err(anyhow::Error::msg)?;
        let _ = self.my_id().await.map_err(anyhow::Error::msg)?;
        println!("Cached {} recent dialogs.", chats.len());
        Ok(())
    }

    /// Phase 1 update handler: logs every incoming message as the XML the LLM
    /// will see later. Phase 2 replaces the body with the notification pipeline.
    pub async fn handle_update(&self, update: Update) {
        if let Update::NewMessage(message) = &update {
            self.on_new_message(message).await;
        }
    }

    async fn on_new_message(&self, message: &grammers_client::message::Message) {
        if message.outgoing() {
            return;
        }
        let cfg = self.config.get().await;
        let my_id = match self.my_id().await {
            Ok(id) => id,
            Err(e) => {
                tracing::error!("cannot resolve own id: {e}");
                return;
            }
        };
        let formatted = llmui::format_history_message(
            &self.tg,
            message,
            my_id,
            &cfg.general.character_name,
            0,
        )
        .await;
        match formatted {
            Ok(xml) => println!("[incoming]\n{xml}\n"),
            Err(e) => tracing::warn!("failed to format message: {e}"),
        }
    }
}
