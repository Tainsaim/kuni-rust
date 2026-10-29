//! Human behavior simulation: rate gating, typing speed, typos, message splitting
//! and anti-repeat checks. Ports of kuni's human-imitation logic
//! (`util/telegramPostMessage`, `send_telegram_message` tool internals).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use rand::Rng;

pub const TYPO_PROBABILITY: f64 = 0.02;
pub const TYPING_DELAY_GRACE: Duration = Duration::from_secs(5);
pub const SINGLE_MESSAGE_MAX_CHARS: usize = 200;

/// Generic error-to-string helper.
pub fn err<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

/// Token-bucket style gate capping the number of sends per 1-second window
/// (kuni throttles: 20 sends per 1s timer tick).
pub struct SendGate {
    sends_per_window: usize,
    state: tokio::sync::Mutex<(Instant, usize)>,
}

impl SendGate {
    pub fn new(sends_per_window: usize) -> Self {
        Self {
            sends_per_window,
            state: tokio::sync::Mutex::new((Instant::now(), 0)),
        }
    }

    /// Waits until a send slot is available.
    pub async fn wait(&self) {
        loop {
            let (window, count) = *self.state.lock().await;
            if count < self.sends_per_window {
                *self.state.lock().await = (window, count + 1);
                return;
            }
            let elapsed = window.elapsed();
            if elapsed >= Duration::from_secs(1) {
                *self.state.lock().await = (Instant::now(), 1);
                return;
            }
            tokio::time::sleep(Duration::from_millis(1000 - elapsed.as_millis().min(999) as u64)).await;
        }
    }
}

/// Estimated time a human would take to type `chars` characters at `wpm`.
pub fn typing_delay(chars: usize, wpm: f64) -> Duration {
    const MIN_DELAY: Duration = Duration::from_millis(250);
    const MAX_DELAY: Duration = Duration::from_secs(20);
    let cps = (wpm * 5.0) / 60.0;
    let secs = chars as f64 / cps.max(0.5);
    Duration::from_secs_f64(secs).clamp(MIN_DELAY, MAX_DELAY)
}

/// Random typing speed within the configured WPM range.
pub fn random_wpm(min: i32, max: i32, rng: &mut impl Rng) -> f64 {
    if max <= min {
        return (min.max(1)) as f64;
    }
    rng.random_range(min as f64..=(max as f64 + 1.0)).clamp(1.0, 400.0)
}

/// Basic Cyrillic detection (no std method for this).
fn is_cyrillic(c: char) -> bool {
    matches!(c, '\u{0410}'..='\u{044F}' | '\u{0401}' | '\u{0451}' | '\u{0400}'..='\u{040F}')
}

fn latin_neighbors(c: &str) -> &'static [&'static str] {
    match c {
        "q" => &["w", "a"],
        "w" => &["q", "e", "s", "a"],
        "e" => &["w", "r", "d", "s"],
        "r" => &["e", "t", "f", "d"],
        "t" => &["r", "y", "g", "f"],
        "y" => &["t", "u", "h", "g"],
        "u" => &["y", "i", "j", "h"],
        "i" => &["u", "o", "k", "j"],
        "o" => &["i", "p", "l", "k"],
        "p" => &["o", "l"],
        "a" => &["q", "w", "s", "z"],
        "s" => &["a", "d", "x", "z", "w", "e"],
        "d" => &["s", "f", "c", "x", "e", "r"],
        "f" => &["d", "g", "v", "c", "r", "t"],
        "g" => &["f", "h", "b", "v", "t", "y"],
        "h" => &["g", "j", "n", "b", "y", "u"],
        "j" => &["h", "k", "m", "n", "u", "i"],
        "k" => &["j", "l", "m", "i", "o"],
        "l" => &["k", "o", "p"],
        "z" => &["a", "x", "s"],
        "x" => &["z", "c", "s", "d"],
        "c" => &["x", "v", "d", "f"],
        "v" => &["c", "b", "f", "g"],
        "b" => &["v", "n", "g", "h"],
        "n" => &["b", "m", "h", "j"],
        "m" => &["n", "j", "k"],
        _ => &[],
    }
}

fn cyrillic_neighbors(c: char) -> &'static [&'static str] {
    match c {
        'й' => &["ц", "ф"],
        'ц' => &["й", "у", "ы", "ф"],
        'у' => &["ц", "к", "в", "ы"],
        'к' => &["у", "е", "а", "в"],
        'е' => &["к", "н", "п", "а"],
        'н' => &["е", "г", "р", "п"],
        'г' => &["н", "ш", "о", "р"],
        'ш' => &["г", "щ", "л", "о"],
        'щ' => &["ш", "з", "д", "л"],
        'з' => &["щ", "х", "ж", "д"],
        'х' => &["з", "ъ", "э", "ж"],
        'ъ' => &["х", "э"],
        'ф' => &["й", "ы", "я", "ц", "а"],
        'ы' => &["ф", "в", "п", "и", "с", "а", "я", "ц", "у"],
        'в' => &["ы", "а", "п", "с", "у", "к", "е", "м"],
        'а' => &["ф", "п", "м", "и", "я", "в", "ы"],
        'п' => &["в", "р", "м", "а", "е", "н", "и", "о"],
        'р' => &["п", "о", "и", "т", "е", "н", "ш"],
        'о' => &["р", "т", "ш", "л", "п", "г"],
        'л' => &["о", "ь", "д", "ш", "щ"],
        'д' => &["л", "ж", "б", "щ", "ч"],
        'ж' => &["д", "э", "ю", "ч"],
        'э' => &["ж", "ю", "х", "ъ"],
        'я' => &["ф", "ч", "с", "а", "ы"],
        'ч' => &["я", "с", "б", "и", "ц", "д", "ж"],
        'с' => &["ы", "и", "т", "в", "я", "ч"],
        'и' => &["с", "т", "ь", "б", "м", "в", "ч"],
        'т' => &["с", "ь", "б", "и", "у", "е"],
        'ь' => &["т", "б", "о", "и"],
        'б' => &["ь", "о", "ю", "н", "д"],
        'ю' => &["б", "э", "о", "ж", "д"],
        _ => &[],
    }
}

/// Keyboard-neighbour replacement ("the finger slipped to the wrong key").
fn random_neighbor_replacement(word: &str, rng: &mut impl Rng) -> String {
    let chars: Vec<char> = word.chars().collect();
    if chars.is_empty() {
        return word.to_string();
    }
    for _ in 0..4 {
        let idx = rng.random_range(0..chars.len());
        let original = chars[idx];
        let lower = original
            .to_lowercase()
            .next()
            .unwrap_or(original);
        let pool: &[&str] = if is_cyrillic(lower) {
            cyrillic_neighbors(lower)
        } else {
            latin_neighbors(&lower.to_string())
        };
        if pool.is_empty() {
            continue;
        }
        let replacement = pool[rng.random_range(0..pool.len())];
        let replacement = if original.is_uppercase() {
            replacement.to_uppercase()
        } else {
            replacement.to_string()
        };
        let mut out = String::with_capacity(word.len() + 2);
        for (i, ch) in chars.iter().enumerate() {
            if i == idx {
                out.push_str(&replacement);
            } else {
                out.push(*ch);
            }
        }
        return out;
    }
    word.to_string()
}

/// Transpose two adjacent letters inside a word.
fn adjacent_transpose(word: &str, rng: &mut impl Rng) -> String {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 4 {
        return word.to_string();
    }
    let idx = rng.random_range(1..chars.len() - 1);
    let mut out = chars;
    out.swap(idx, idx + 1);
    out.into_iter().collect()
}

/// Applies the kuni typo simulation: with `TYPO_PROBABILITY` per message,
/// one word is replaced by a keyboard neighbor or slightly shuffled.
pub fn apply_typo(text: &str, rng: &mut impl Rng) -> String {
    if !rng.random_bool(TYPO_PROBABILITY) {
        return text.to_string();
    }

    // Collect candidate tokens (>= 4 chars) with their byte positions.
    let mut candidates: Vec<(usize, &str)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                if text[s..i].chars().count() >= 4 {
                    candidates.push((s, &text[s..i]));
                }
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        if text[s..].chars().count() >= 4 {
            candidates.push((s, &text[s..]));
        }
    }
    if candidates.is_empty() {
        return text.to_string();
    }

    let (idx, word) = candidates[rng.random_range(0..candidates.len())];
    let replaced = loop {
        let candidate = if rng.random_bool(0.5) {
            random_neighbor_replacement(word, rng)
        } else {
            adjacent_transpose(word, rng)
        };
        if candidate != word {
            break candidate;
        }
    };
    format!("{}{}{}", &text[..idx], replaced, &text[idx + word.len()..])
}

/// Splits an LLM message into per-line messages (one logical line each,
/// hard-splitting long lines — kuni's "multiple short messages 1 line each" rule).
pub fn split_for_sending(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if line.chars().count() <= SINGLE_MESSAGE_MAX_CHARS {
            out.push(line.to_string());
            continue;
        }
        let mut rest: Vec<char> = line.chars().collect();
        while rest.len() > SINGLE_MESSAGE_MAX_CHARS {
            let mut split_at = SINGLE_MESSAGE_MAX_CHARS;
            for i in (1..SINGLE_MESSAGE_MAX_CHARS).rev() {
                if rest[i].is_whitespace() {
                    split_at = i + 1;
                    break;
                }
            }
            let head: String = rest[..split_at].iter().collect();
            if !head.trim().is_empty() {
                out.push(head.trim_end().to_string());
            }
            rest = rest.split_off(split_at);
        }
        if !rest.is_empty() {
            let tail: String = rest.into_iter().collect();
            if !tail.trim().is_empty() {
                out.push(tail.trim_end().to_string());
            }
        }
    }
    out
}

/// Anti-repeat store — kuni's anti-repetition for own messages using embeddings
/// (`anti_repeat_trigger_max/avg`, +0.07 head start soak per offender).
#[derive(Default)]
pub struct AntiRepeat {
    embedding_history: VecDeque<Vec<f64>>,
    offender_head_start: f64,
}

impl AntiRepeat {
    pub fn push(&mut self, embedding: Option<Vec<f64>>, max_history: usize) {
        if let Some(embedding) = embedding {
            self.embedding_history.push_front(embedding);
            self.embedding_history.truncate(max_history);
        }
    }

    /// Returns an error line for the LLM if the message would repeat previous ones.
    pub fn violation(
        &mut self,
        embedding: Option<&[f64]>,
        trigger_max: f64,
        trigger_avg: f64,
    ) -> Option<String> {
        let embedding = embedding?;
        if self.embedding_history.is_empty() {
            self.offender_head_start = (self.offender_head_start - 0.07).max(0.0);
            return None;
        }
        let mut sims: Vec<f64> = self
            .embedding_history
            .iter()
            .map(|e| cosine_similarity(embedding, e))
            .collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let max = sims[0];
        let avg_slice_len = sims.len().saturating_sub(6).max(1);
        let avg = if sims.len() > 6 {
            sims[6..].iter().sum::<f64>() / avg_slice_len as f64
        } else {
            sims.iter().sum::<f64>() / sims.len() as f64
        };
        let head = self.offender_head_start;
        if max > trigger_max - head || avg > trigger_avg - head.min(0.2) {
            self.offender_head_start = (head + 0.07).min(0.4);
            Some(format!(
                "Too similar to previous messages (max {max:.3} > {:.3}, avg {avg:.3}). \
                 Don't repeat yourself. Consider calling #wait instead, or send a reaction, \
                 or ask the diary (#ask).",
                trigger_max - head
            ))
        } else {
            self.offender_head_start = (self.offender_head_start - 0.07).max(0.0);
            None
        }
    }
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Per-chat send streak (`messagesInARow` in kuni: >5 warn, >10 fail).
pub struct SendStreak {
    pub in_row: u32,
    pub last: Instant,
}

impl SendStreak {
    pub fn new() -> Self {
        Self {
            in_row: 0,
            last: Instant::now() - Duration::from_secs(600),
        }
    }
}

/// Decayed-reset helper: a streak older than 10 minutes restarts from zero.
pub fn streak_decay(streak: &mut SendStreak) {
    if streak.last.elapsed() > Duration::from_secs(600) {
        streak.in_row = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn typo_until(rng_seed: u64, text: &str, tries: usize) -> String {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        for _ in 0..tries {
            let out = apply_typo(text, &mut rng);
            if out != text {
                return out;
            }
        }
        text.to_string()
    }

    #[test]
    fn split_respects_line_boundary() {
        let parts = split_for_sending("привет\nчто делаешь?");
        assert_eq!(parts, vec!["привет", "что делаешь?"]);
    }

    #[test]
    fn split_long_lines() {
        let long = format!("{}{}", "а".repeat(350), " б".repeat(50));
        let parts = split_for_sending(&long);
        assert!(parts.iter().all(|p| p.chars().count() <= SINGLE_MESSAGE_MAX_CHARS));
        assert!(parts.len() >= 2);
    }

    #[test]
    fn typing_delay_scales() {
        let slow = typing_delay(100, 120.0);
        let fast = typing_delay(100, 300.0);
        assert!(slow > fast);
    }

    #[test]
    fn typos_sometimes_apply() {
        let russian = "привет как дела у тебя сегодня";
        let mutated = typo_until(1234567890, russian, 400);
        assert_ne!(mutated, russian, "typo should eventually apply");
        let latin = "hello world this is a test string for typos";
        let mutated = typo_until(987654321, latin, 400);
        assert_ne!(mutated, latin);
    }

    #[test]
    fn anti_repeat_works() {
        let mut anti = AntiRepeat::default();
        anti.push(Some(vec![1.0, 0.0]), 32);
        anti.push(Some(vec![1.0, 0.0]), 32);
        let v = anti.violation(Some(&[1.0, 0.0]), 0.95, 0.85);
        assert!(v.is_some(), "identical embedding must trigger anti-repeat");
        let fresh = AntiRepeat::default();
        let mut fresh = fresh;
        assert!(fresh.violation(None, 0.95, 0.85).is_none());
    }

    #[test]
    fn cosine_similarity_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
    }
}
