//! LLM-bound chat history formatting, a port of kuni's `src/llmui/telegram.cpp`.

pub mod dialog;
pub mod format;

pub use format::format_history_message;
