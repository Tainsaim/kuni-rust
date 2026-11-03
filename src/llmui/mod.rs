//! LLM-bound chat history formatting, a port of kuni's `src/llmui/telegram.cpp`.

pub mod dialog;
pub mod format;

pub use format::format_history_message;

/// kuni's `llmui::checkForMaliciousPayloads` (malicious_payloads.cpp): all strings
/// that come to the LLM from outside (message contents, user names and everything
/// else) must be checked first; if a violation is caused, the string is replaced
/// with "malicious". The violation is smuggling kuni's inline-image embedding tag,
/// which would weaponize the worker's low-quality tool-call rollback.
pub fn check_malicious_payloads(string: &mut String) {
    if string.contains(crate::llm::EMBEDDING_TAG) {
        *string = "malicious".to_string();
    }
}
