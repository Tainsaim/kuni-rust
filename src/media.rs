//! Media understanding (vision/hearing input capability): photos and stickers are
//! described by the `llm_image_to_text` model, voice notes are transcribed by the
//! `llm_audio_to_text` model. Ports of kuni's `src/llmui/image.cpp` and
//! `src/llmui/audio.cpp` minus the video pipeline (ffmpeg frame extraction;
//! `llmui::video` / `telegram_video_messages` is not ported — media stays as the
//! plain `[video]` / `[video note]` / `[animation]` markers) and minus the Telegram
//! Premium `recognizeSpeech` path (its `updateMessageContent` subscription does not
//! fit our event flow; we go straight to the whisper fallback, kuni's fallback too).

use std::sync::Arc;

use base64::Engine as _;

use crate::app::App;
use crate::llm::{self, Message};

/// kuni's `image.resizedLinearDownscale({672, 672})` box.
const MAX_EMBED_SIZE: u32 = 672;
/// kuni's `goto tryAgain` busy loop retries endlessly on empty content; like the
/// diary consolidation port, we cap the attempts (see llmui/image.cpp quirk note).
const MAX_PARSE_ATTEMPTS: u32 = 3;
/// kuni's `llmui::image` catch-all body.
const NOT_SUPPORTED: &str = "This media type is not supported";

fn not_supported(xml_tag: &str) -> String {
    format!("<{xml_tag} description>\n{NOT_SUPPORTED}\n</{xml_tag}>")
}

/// kuni's `llmui::image`: caches by content hash (kuni keys by the td-local file
/// name of the download, which is session-bound; a content hash survives restarts),
/// re-encodes to a fit-672 JPEG, embeds it into the captioning context and asks
/// `llm_image_to_text` for the description. Always returns a string: on any
/// failure the kuni "not supported" text.
pub async fn describe_image(
    app: &Arc<App>,
    context_items: &[Message],
    image_bytes: &[u8],
    xml_tag: &str,
) -> String {
    // Cache dir `cache/images` relative to the working directory, like kuni's
    // `APath("cache") / "images"`.
    let cache_dir = std::path::Path::new("cache").join("images");
    if tokio::fs::create_dir_all(&cache_dir).await.is_err() {
        // Cache is best-effort; proceed without it.
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&image_bytes, &mut hasher);
    let hash = std::hash::Hasher::finish(&mut hasher);
    let cache_path = cache_dir.join(format!("{hash:016x}.md"));

    if let Ok(cached) = tokio::fs::read_to_string(&cache_path).await {
        return format!("<{xml_tag} description>\n{cached}\n</{xml_tag}>");
    }

    let description = describe_image_uncached(app, context_items, image_bytes, xml_tag).await;
    if let Some(desc) = &description {
        let _ = tokio::fs::write(&cache_path, desc).await;
    }
    description.unwrap_or_else(|| not_supported(xml_tag))
}

async fn describe_image_uncached(
    app: &Arc<App>,
    context_items: &[Message],
    image_bytes: &[u8],
    xml_tag: &str,
) -> Option<String> {
    let cfg = app.config.get().await;
    let Some(emd) = cfg.capabilities.vision.llm_image_to_text.clone() else {
        // kuni with no reachable vision endpoint fails in openAI.chat and lands in
        // the same "not supported" catch; our config without any endpoint goes
        // straight there.
        return None;
    };
    drop(cfg);

    // kuni decodes any format, downscales linearly into the 672x672 box and saves
    // the JPEG (JpgImageLoader) regardless of the source codec.
    let img = image::load_from_memory(image_bytes).ok()?;
    let resized = fit_linear_downscale(&img, MAX_EMBED_SIZE);
    let mut jpeg = std::io::Cursor::new(Vec::new());
    resized
        .write_to(&mut jpeg, image::ImageFormat::Jpeg)
        .ok()?;
    let base64 = base64::engine::general_purpose::STANDARD.encode(jpeg.get_ref());

    // Port of the captioning context verbatim, kuni's broken
    // `</character name="...">` quirk included.
    let vars = app.prompt_vars().await;
    let prompts = app.prompts.get(&vars).await;
    let mut context = String::from("<context>\n");
    let name = vars.character_name;
    context += &format!("<character name=\"{name}\">\n");
    context += &prompts.character_appearance;
    context += &format!("\n</character name=\"{name}\">\n");
    for item in context_items {
        context += "<context_item>\n";
        context += &item.content;
        context += "\n</context_item>\n";
    }
    // kuni routes video extensions into llmui::video at this spot (not ported).
    context += "\n\n</context>\n\nPhoto:\n\n";
    context += &llm::embed_image(&base64);
    context += "\n\nDescribe the last photo.";

    // hack: ask for shorter descriptions for stickers (kuni's comment).
    let prompt = if xml_tag.contains("sticker") {
        &prompts.sticker_to_text
    } else {
        &prompts.photo_to_text
    };

    for attempt in 1..=MAX_PARSE_ATTEMPTS {
        let response = app.llm.chat_once(&emd, Some(prompt), &[Message::user(context.clone())])
            .await
            .map_err(|e| tracing::error!("llmui::image: Can't describe photo {e}"))
            .ok()?;
        // probably a local model overflown its reasoning: retry on empty.
        if !response.content.trim().is_empty() {
            return Some(response.content);
        }
        tracing::warn!("llmui::image: empty description (attempt {attempt}/{MAX_PARSE_ATTEMPTS})");
        if attempt < MAX_PARSE_ATTEMPTS {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }
    // kuni loops forever; we give up with the failure text.
    None
}

/// `AImage::resizedLinearDownscale({size, size})`: scale so that both dimensions
/// fit the box, preserving the aspect ratio (images already inside the box stay).
fn fit_linear_downscale(img: &image::DynamicImage, box_size: u32) -> image::DynamicImage {
    let (w, h) = (img.width().max(1), img.height().max(1));
    if w <= box_size && h <= box_size {
        return img.clone();
    }
    let max_dim = w.max(h);
    let new_w = (f64::from(w) * f64::from(box_size) / f64::from(max_dim)) as u32;
    let new_h = (f64::from(h) * f64::from(box_size) / f64::from(max_dim)) as u32;
    img.resize_exact(new_w.max(1), new_h.max(1), image::imageops::FilterType::Triangle)
}

/// kuni's `llmui::voiceMessage`: transcribes a voice note with the whisper
/// endpoint. kuni gauges on `capabilityHearing` + `llmAudioToText` separately; our
/// config merges the two (the endpoint section presence IS the capability), so an
/// unset `[capabilities.hearing].llm_audio_to_text` means the same as
/// `capabilityHearing = false`: empty string, the `[voice message]` marker stays.
pub async fn transcribe_voice(app: &Arc<App>, voice_bytes: Vec<u8>) -> String {
    if app
        .config
        .get()
        .await
        .capabilities
        .hearing
        .llm_audio_to_text
        .is_none()
    {
        return String::new();
    }
    match app.llm.transcribe_audio(voice_bytes, "ogg").await {
        Ok(text) => format!("<voice>\n{text}\n</voice>"),
        Err(e) => {
            tracing::error!("voiceMessage: Can't transcribe audio {e}");
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_supported_is_kuni_text() {
        let s = not_supported("photo");
        assert_eq!(s, "<photo description>\nThis media type is not supported\n</photo>");
    }

    #[test]
    fn downscale_fits_box() {
        let img = image::DynamicImage::new_rgb8(1344, 672);
        let out = fit_linear_downscale(&img, MAX_EMBED_SIZE);
        assert_eq!(out.width(), MAX_EMBED_SIZE);
        assert_eq!(out.height(), MAX_EMBED_SIZE / 2);

        let small = image::DynamicImage::new_rgb8(300, 200);
        let out = fit_linear_downscale(&small, MAX_EMBED_SIZE);
        // Images inside the box stay untouched.
        assert_eq!(out.width(), 300);
    }
}
