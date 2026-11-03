//! `<chat .../>` listing, a port of kuni's `llmui::formatChatSingle`.

use grammers_client::tl as tl;
use grammers_client::message::Message;
use grammers_client::peer::Peer;

use super::format::ago;

fn type_name(peer: &Peer) -> &'static str {
    match peer {
        Peer::User(_) => "direct messages",
        Peer::Group(_) => "group chat",
        Peer::Channel(_) => "channel",
        _ => "unknown",
    }
}

fn escape(s: &str) -> String {
    s.replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Truncates in the middle so that the head and tail of the string are intact
/// (kuni truncates previews to ~80 chars, middle-out).
fn middle_truncate(s: &str, max_chars: usize) -> String {
    let len = s.chars().count();
    if len <= max_chars {
        return s.to_string();
    }
    let head = max_chars * 2 / 3;
    let tail = max_chars.saturating_sub(head);
    let mut out = String::new();
    out.extend(s.chars().take(head));
    out.push('…');
    out.extend(s.chars().skip(len.saturating_sub(tail)));
    out
}

/// Best-effort chat/sender title for a cached peer.
pub fn peer_title(peer: &Peer) -> String {
    let mut title = match peer {
        Peer::User(user) => user.full_name(),
        Peer::Group(group) => group.title().unwrap_or("unnamed group").to_string(),
        Peer::Channel(channel) => channel.title().to_string(),
        _ => "unknown".to_string(),
    };
    super::check_malicious_payloads(&mut title);
    title
}

/// Single dialog as the LLM sees it:
/// `<chat chat_id="…" title="…" preview="{sender}: {text}" type="…" last_message="{x ago}" [unread_count="N"] />`
pub fn format_chat_single(
    dialog_id: i64,
    peer: &Peer,
    unread_count: u32,
    last_message: Option<&Message>,
    now: jiff::Timestamp,
) -> String {
    let title = match peer {
        Peer::User(user) => user.full_name(),
        Peer::Group(group) => group.title().unwrap_or("unnamed group").to_string(),
        Peer::Channel(channel) => channel.title().to_string(),
        _ => "unknown".to_string(),
    };
    let mut line = format!(
        "<chat chat_id=\"{dialog_id}\" title=\"{}\" type=\"{}\"",
        escape(&title),
        type_name(peer)
    );
    if let Some(message) = last_message {
        let from = message
            .sender_id()
            .and_then(|i| i.bot_api_dialog_id())
            .unwrap_or(0)
            .to_string();
        let mut text = message.text().to_string();
        super::check_malicious_payloads(&mut text);
        let preview = middle_truncate(&text, 80);
        line.push_str(&format!(
            " preview=\"{}\" last_message=\"{}\"",
            escape(&format!("{from}: {preview}")),
            escape(&ago(message.date()))
        ));
    } else {
        line.push_str(&format!(" last_message=\"{}\"", escape(&ago(now))));
    }
    if unread_count > 0 {
        line.push_str(&format!(" unread_count=\"{unread_count}\""));
    }
    format!("{line} />")
}

/// Unread count of a dialog, decoded from the raw TL dialog, or 0.
pub fn dialog_unread(raw: &tl::enums::Dialog) -> u32 {
    match raw {
        tl::enums::Dialog::Dialog(dialog) => dialog.unread_count.max(0) as u32,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_truncate_keeps_ends() {
        let s = middle_truncate(&"х".repeat(200), 80);
        assert!(s.chars().count() < 85);
        assert!(s.starts_with("ххх"));
        assert!(s.ends_with("ххх"));
    }
}
