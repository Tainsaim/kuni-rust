//! `<message .../>` XML formatting of chat history, a port of kuni's
//! `llmui::formatChatHistoryMessage` (`src/llmui/telegram.cpp`).

use grammers_client::grammers_tl_types as tl;
use grammers_client::message::{Media, Message};
use grammers_client::peer::Peer;

use crate::tg::Tg;

/// Formats a timestamp the way the LLM reads it: HH:MM for today, MM-DD HH:MM otherwise.
fn format_date(timestamp: jiff::Timestamp) -> String {
    let tz = jiff::tz::TimeZone::system();
    let zoned = timestamp.to_zoned(tz);
    let now = jiff::Zoned::now();
    if zoned.year() == now.year() && zoned.month() == now.month() && zoned.day() == now.day() {
        format!("{:02}:{:02}", zoned.hour(), zoned.minute())
    } else {
        format!(
            "{:02}-{:02} {:02}:{:02}",
            zoned.month(),
            zoned.day(),
            zoned.hour(),
            zoned.minute()
        )
    }
}

/// Extracts a human-readable text representation of message media, or `None` for pure text.
pub fn media_marker(media: &Media) -> String {
    match media {
        Media::Photo(_) => "[photo]".to_string(),
        Media::Sticker(sticker) => {
            format!("<sticker emoji=\"{}\" />", html_escape(sticker.emoji()))
        }
        Media::Dice(dice) => format!("[dice {}]", html_escape(dice.emoji())),
        Media::Geo(_) | Media::GeoLive(_) | Media::Venue(_) => "[location]".to_string(),
        Media::Poll(_) => "[poll]".to_string(),
        Media::Contact(_) => "[contact]".to_string(),
        Media::WebPage(_) => "[web page]".to_string(),
        // Documents: distinguish voice notes, video notes, animations and plain files.
        Media::Document(doc) => {
            let attrs = match &doc.raw.document {
                Some(tl::enums::Document::Document(document)) => &document.attributes,
                _ => return "[file]".to_string(),
            };
            let mut animated = false;
            let mut voice = false;
            let mut video = false;
            let mut video_note = false;
            let mut sticker = false;
            for attr in attrs {
                match attr {
                    tl::enums::DocumentAttribute::Animated(_) => animated = true,
                    tl::enums::DocumentAttribute::Audio(audio) => voice = audio.voice,
                    tl::enums::DocumentAttribute::Video(video) => video_note = video.video_note,
                    tl::enums::DocumentAttribute::Sticker(_) => sticker = true,
                    _ => {}
                }
            }
            if sticker {
                "[sticker]".to_string()
            } else if voice {
                "[voice message]".to_string()
            } else if video_note {
                "[video note]".to_string()
            } else if animated {
                "[animation]".to_string()
            } else if video {
                "[video]".to_string()
            } else {
                match doc.raw.file_name.as_ref().map(|f| f.as_str()) {
                    Some(name) => format!("[file {}]", html_escape(name)),
                    None => "[file]".to_string(),
                }
            }
        }
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Sender display name: for users first+last name, for groups/channels the title.
fn sender_name(peer: &Peer) -> String {
    match peer {
        Peer::User(user) => {
            let name = user.full_name();
            if name.is_empty() {
                "unnamed".to_string()
            } else {
                name
            }
        }
        Peer::Group(group) => group
            .title()
            .map(|t| t.to_string())
            .unwrap_or_else(|| "unnamed group".to_string()),
        Peer::Channel(channel) => channel.title().to_string(),
        _ => "unnamed".to_string(),
    }
}

/// One chat history message as the LLM sees it, recursively prepending the replied-to
/// message (kuni's recursive `reply_to` formatting).
pub async fn format_history_message(
    tg: &Tg,
    message: &Message,
    my_id: i64,
    character_name: &str,
    depth: u8,
) -> Result<String, String> {
    if depth > 5 {
        // Guard against pathological reply chains.
        return Err("reply chain too deep".to_string());
    }
    let sender_id: i64 = message
        .sender_id()
        .and_then(|id| id.bot_api_dialog_id())
        .unwrap_or(0);
    let mut line = String::new();
    line.push_str("<message");
    line.push_str(&format!(" message_id={}", xml_attr(message.id().to_string())));
    line.push_str(&format!(
        " date={}",
        xml_attr(format_date(message.date()))
    ));
    // Sender: own messages show as the character itself; forwards show origin + sharer.
    if let Some(forward_header) = message.forward_header() {
        if let tl::enums::MessageFwdHeader::Header(header) = forward_header {
            if let Some(from_id) = header.from_id {
                let id = peer_ref_id(&from_id);
                if let Ok(id) = id {
                    line.push_str(&format!(
                        " sender={} forwarded_by={}",
                        xml_attr(entity_name(tg, id).await),
                        xml_attr(
                            if sender_id == my_id {
                                character_name.to_string()
                            } else {
                                entity_name(tg, sender_id).await
                            }
                        )
                    ));
                }
            }
        }
    } else if sender_id == my_id {
        line.push_str(&format!(" sender={}", xml_attr(character_name.to_string())));
    } else if let Some(sender) = message.sender() {
        let id_attr = if matches!(sender, Peer::User(_)) {
            format!(" sender_id={}", xml_attr(sender_id.to_string()))
        } else {
            String::new()
        };
        line.push_str(&format!(
            " sender={}{id_attr}",
            xml_attr(sender_name(sender))
        ));
    }
    line.push_str(">");
    line.push_str("\n");
    // Replied-to message prepended.
    if message.reply_to_message_id().is_some() {
        if depth < 5 {
            if let Ok(Some(reply)) = message.get_reply().await {
                if let Ok(formatted) =
                    Box::pin(format_history_message(tg, &reply, my_id, character_name, depth + 1))
                        .await
                {
                    line.push_str(&formatted);
                }
            } else {
                line.push_str("<reply_to>Deleted Message</reply_to>\n");
            }
        } else {
            line.push_str("<reply_to>Deleted Message</reply_to>\n");
        }
    }
    // Reactions display (count only for now: emoji pairing needs raw interaction data).
    if let Some(count) = message.reaction_count() {
        line.push_str(&format!("<reactions count=\"{count}\" />\n"));
    }
    // Body: markdown-ish view of the text if present, else media marker.
    let text = message.text();
    let media = message.media();
    let body = if !text.is_empty() {
        Some(text.to_string())
    } else {
        media.as_ref().map(media_marker)
    };
    if let Some(body) = body {
        line.push_str(&html_escape(&body));
        line.push('\n');
    }
    // Post author signature etc. for channels.
    if let Some(author) = message.post_author() {
        line.push_str(&format!("<post_author>{}</post_author>\n", html_escape(author)));
    }
    line.push_str("</message>");
    Ok(line)
}

fn peer_ref_id(
    peer: &tl::enums::Peer,
) -> Result<i64, String> {
    match peer {
        tl::enums::Peer::PeerUser(user) => Ok(user.user_id),
        tl::enums::Peer::PeerChat(chat) => Ok(-chat.chat_id),
        tl::enums::Peer::PeerChannel(channel) => {
            Ok(-1000000000000 - channel.channel_id)
        }
    }
}

/// Best-effort sender name by id using the runtime peer cache.
async fn entity_name(tg: &Tg, dialog_id: i64) -> String {
    match tg.chat(dialog_id).await {
        Some(peer) => sender_name(&peer),
        None => format!("id {dialog_id}"),
    }
}

fn xml_attr(value: String) -> String {
    format!("\"{}\"", html_escape(&value))
}

/// Human "N minutes ago" formatting for dialog previews.
pub fn ago(timestamp: jiff::Timestamp) -> String {
    let now = jiff::Timestamp::now();
    let secs = now.as_second() - timestamp.as_second();
    let secs = secs.max(0);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        let mins = secs / 60;
        if mins == 1 {
            "1 minute ago".to_string()
        } else {
            format!("{mins} minutes ago")
        }
    } else if secs < 86400 {
        let hours = secs / 3600;
        if hours == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{hours} hours ago")
        }
    } else {
        let days = secs / 86400;
        if days == 1 {
            "yesterday".to_string()
        } else {
            format!("{days} days ago")
        }
    }
}
