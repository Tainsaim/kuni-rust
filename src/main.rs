//! Entry point: config bootstrap, Telegram connection and the Phase 1 smoke run.

mod app;
mod config;
mod llmui;
mod tg;

use anyhow::Result;
use grammers_client::Client;
use grammers_mtsender::{SenderPool, UpdatesConfiguration};

const SESSION_FILE: &str = "kuni.session";

fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    let config = config::ConfigHub::bootstrap(std::path::Path::new("config.toml")).await?;
    let cfg = config.get().await;

    let (tg, pool) = tg::Tg::setup(SESSION_FILE, cfg.general.telegram_api_id).await?;
    let SenderPool {
        runner,
        updates,
        handle,
    } = pool;
    let pool_task = tokio::spawn(runner.run());

    if let Err(e) = tg.authorize_interactive(&cfg.general.telegram_api_hash).await {
        anyhow::bail!("authorization failed: {e}");
    }
    tg.set_online(true).await.map_err(anyhow::Error::msg)?;
    println!("Online. Waiting for messages (Ctrl+C to quit)...");

    let app = app::App::new(config, tg.clone(), handle.clone()).await?;
    app.init().await?;
    app.refresh_config().await;

    let client = Client::new(handle.clone());
    let mut stream = client
        .stream_updates(updates, UpdatesConfiguration { catch_up: true })
        .await
        .map_err(|e| anyhow::anyhow!("stream updates failed: {e}"))?;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            update = stream.next() => {
                match update {
                    Ok(update) => app.handle_update(update).await,
                    Err(e) => {
                        tracing::error!("update stream error: {e}");
                        break;
                    }
                }
            }
        }
    }

    println!("Shutting down...");
    tg.set_online(false).await.ok();
    stream.sync_update_state().await.ok();
    handle.quit();
    let _ = pool_task.await;
    println!("Done.");
    Ok(())
}
