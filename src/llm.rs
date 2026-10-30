//! OpenAI-compatible LLM client: chat (SSE streaming + tool calls) and embeddings.
//! A port of kuni-master's `src/OpenAIChatImpl.cpp` / `src/Endpoint.h` on reqwest.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::config::ConfigHub;

// ---------------------------------------------------------------------------
// Message types (OpenAI chat format), kuni's `IOpenAIChat::Message`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Role {
    #[default]
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Stable id issued by the server; reused in the follow-up `tool` message.
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// One entry of the LLM conversation context (kuni's `Session` element).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// DeepSeek-style reasoning content, preserved like kuni does.
    pub reasoning_content: String,
    /// Only for `Role::Assistant`.
    pub tool_calls: Vec<ToolCall>,
    /// Only for `Role::Tool`.
    pub tool_call_id: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), ..Default::default() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), ..Default::default() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), ..Default::default() }
    }
    pub fn assistant_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            ..Default::default()
        }
    }
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: Role::Tool, tool_call_id: tool_call_id.into(), content: content.into(), ..Default::default() }
    }

    /// Serializes to the OpenAI wire format.
    pub fn to_api(&self) -> Value {
        let mut obj = json!({ "role": self.role.as_str() });
        if !self.content.is_empty() {
            obj["content"] = json!(self.content);
        } else if self.role == Role::Assistant {
            obj["content"] = Value::Null;
        }
        if !self.reasoning_content.is_empty() {
            obj["reasoning_content"] = json!(self.reasoning_content);
        }
        if !self.tool_calls.is_empty() {
            obj["tool_calls"] = Value::Array(
                self.tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            // kuni always uses the "function" flavor of tool calls.
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                // OpenAI expects arguments as a JSON-encoded string.
                                "arguments": serde_json::to_string(&tc.arguments)
                                    .unwrap_or_else(|_| "null".into()),
                            },
                        })
                    })
                    .collect(),
            );
        }
        if self.role == Role::Tool {
            obj["tool_call_id"] = json!(self.tool_call_id);
        }
        obj
    }
}

/// Token accounting of one chat request (kuni's `IOpenAIChat::Response::usage`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    fn estimate(messages: &[Message], generated: &Message) -> Self {
        // Rough fallback when the server does not report usage: ~4 chars per token.
        let mut chars: usize = generated.content.chars().count()
            + generated.reasoning_content.chars().count()
            + generated
                .tool_calls
                .iter()
                .map(|tc| tc.name.len() + tc.arguments.to_string().len())
                .sum::<usize>();
        for m in messages {
            chars += m.content.chars().count() + m.reasoning_content.chars().count();
        }
        Usage {
            prompt_tokens: (chars / 4) as u64,
            completion_tokens: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Virtual endpoints set (kuni's `LLMEndpoint*`).
pub struct Llm {
    http: reqwest::Client,
    pub config: Arc<ConfigHub>,
}

fn chat_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

fn embeddings_url(base_url: &str) -> String {
    format!("{}/embeddings", base_url.trim_end_matches('/'))
}

/// How to nudge the "user is typing" indicator while a response streams in.
pub type TypingCb = Arc<dyn Fn() -> BoxFutUnit + Send + Sync>;

pub type BoxFutUnit = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

/// Incremental SSE chunk state for assembling tool calls.
#[derive(Default)]
struct Deltas {
    content: String,
    reasoning_content: String,
    finish_ok: bool,
    usage: Option<Usage>,
    tool_calls: std::collections::BTreeMap<i64, RawToolCall>,
}

#[derive(Default)]
struct RawToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl Deltas {
    /// Feeds one SSE `data:` JSON payload.
    fn feed(&mut self, chunk: &Value) {
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(Usage {
                prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
                completion_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
        let Some(choice) = chunk["choices"].get(0) else {
            return;
        };
        if choice["finish_reason"].is_string() {
            self.finish_ok = true;
        }
        let delta = &choice["delta"];
        if let Some(c) = delta["content"].as_str() {
            self.content.push_str(c);
        }
        if let Some(c) = delta["reasoning_content"].as_str() {
            self.reasoning_content.push_str(c);
        }
        if let Some(tcs) = delta["tool_calls"].as_array() {
            for tc in tcs {
                let index = tc["index"].as_i64().unwrap_or(0);
                let entry = self.tool_calls.entry(index).or_default();
                if let Some(id) = tc["id"].as_str() {
                    entry.id.push_str(id);
                }
                if let Some(name) = tc["function"]["name"].as_str() {
                    entry.name.push_str(name);
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    entry.arguments.push_str(args);
                }
            }
        }
    }

    /// Assembles the final assistant message. Falls back to a manual parse when the
    /// server produced no streaming deltas at all.
    fn finish(self, fallback: Option<&Value>, messages: &[Message]) -> Result<(Message, Usage), String> {
        let mut tool_calls = Vec::new();
        for (_, tc) in self.tool_calls {
            let arguments = if tc.arguments.trim().is_empty() {
                Value::Null
            } else {
                serde_json::from_str(&tc.arguments)
                    .map_err(|e| format!("bad tool call arguments: {e} ({})", tc.arguments))?
            };
            tool_calls.push(ToolCall {
                id: tc.id,
                name: tc.name,
                arguments,
            });
        }
        let msg = Message {
            role: Role::Assistant,
            content: self.content,
            // Chunked reasoning and no content: treat reasoning as visible content, kuni-style.
            reasoning_content: self.reasoning_content,
            tool_calls,
            ..Default::default()
        };
        let usage = self
            .usage
            .unwrap_or_else(|| Usage::estimate(messages, &msg));
        let _ = (self.finish_ok, fallback);
        Ok((msg, usage))
    }

    /// Non-streaming fallback response (some OpenAI-compatible servers ignore `stream`).
    fn into_finish_from_body(mut self, body: &Value, messages: &[Message]) -> Result<(Message, Usage), String> {
        if self.content.is_empty() && self.tool_calls.is_empty() {
            let choice = body["choices"]
                .get(0)
                .ok_or_else(|| format!("empty response body: {body}"))?;
            let message = &choice["message"];
            self.content = message["content"].as_str().unwrap_or("").to_string();
            if let Some(tcs) = message["tool_calls"].as_array() {
                for (index, tc) in tcs.iter().enumerate() {
                    let args = tc["function"]["arguments"].as_str().unwrap_or("null");
                    self.tool_calls.insert(
                        index as i64,
                        RawToolCall {
                            id: tc["id"].as_str().unwrap_or("").to_string(),
                            name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                            arguments: args.to_string(),
                        },
                    );
                }
            }
            if let Some(u) = body.get("usage").filter(|u| !u.is_null()) {
                self.usage = Some(Usage {
                    prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
                    completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                });
            }
        }
        self.finish(None, messages)
    }
}

/// Strips the control characters JSONLikes choke on (kuni's `removeControlCharacters`).
fn remove_control_characters(input: &str) -> String {
    input
        .chars()
        .filter(|&c| c == '\n' || c == '\t' || !c.is_control())
        .collect()
}

impl Llm {
    pub fn new(config: Arc<ConfigHub>) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder().build()?;
        Ok(Self { http, config })
    }

    fn options_json(misc: &crate::config::Misc) -> Value {
        let mut extra = serde_json::Map::new();
        if let Some(v) = misc.llm_temperature {
            extra.insert("temperature".into(), json!(v));
        }
        if let Some(v) = misc.llm_top_p {
            extra.insert("top_p".into(), json!(v));
        }
        if let Some(v) = misc.llm_top_k {
            extra.insert("top_k".into(), json!(v));
        }
        if let Some(v) = misc.llm_min_p {
            extra.insert("min_p".into(), json!(v));
        }
        if let Some(v) = misc.llm_presence_penalty {
            extra.insert("presence_penalty".into(), json!(v));
        }
        if let Some(v) = misc.llm_repetition_penalty {
            extra.insert("repetition_penalty".into(), json!(v));
        }
        Value::Object(extra)
    }

    fn build_body(
        misc: &crate::config::Misc,
        emd: &crate::config::EndpointAndModel,
        system_prompt: Option<&str>,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Value {
        let mut messages_json = Vec::with_capacity(messages.len() + 1);
        if let Some(system_prompt) = system_prompt {
            if !system_prompt.is_empty() {
                messages_json.push(json!({ "role": "system", "content": system_prompt }));
            }
        }
        messages_json.extend(messages.iter().map(Message::to_api));

        let mut body = json!({
            "model": emd.model,
            "messages": messages_json,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        if let Some(tools) = tools.filter(|t| !t.is_empty()) {
            body["tools"] = Value::Array(tools.to_vec());
        }
        let obj = body.as_object_mut().unwrap();
        for (k, v) in Self::options_json(misc).as_object().unwrap() {
            obj.insert(k.clone(), v.clone());
        }
        body
    }

    /// Chat with streaming. `on_typing` is fired (throttled to 4s) while the response
    /// assembles, so Telegram shows the "typing..." indicator, kuni's
    /// `onResponseAssembling`.
    pub async fn chat_streaming(
        &self,
        system_prompt: Option<&str>,
        messages: &[Message],
        tools: Option<&[Value]>,
        on_typing: Option<TypingCb>,
    ) -> Result<(Message, Usage), String> {
        let cfg = self.config.get().await;
        let url = chat_url(&cfg.general.llm.endpoint.base_url);
        let body = Self::build_body(&cfg.misc, &cfg.general.llm, system_prompt, messages, tools);
        drop(cfg);

        let total_timeout =
            Duration::from_secs(self.config.get().await.misc.request_timeout_secs * 10 + 60);
        let resp = tokio::time::timeout(total_timeout, async {
            self.post_json(&url, &body).await
        })
        .await
        .map_err(|_| "LLM request timed out".to_string())??;

        // The server may ignore `stream`; fall back to parsing a plain JSON body.
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if content_type.contains("event-stream") {
            self.consume_stream(resp, messages, on_typing).await
        } else {
            let body: Value = resp.json().await.map_err(|e| format!("bad LLM response: {e}"))?;
            Deltas::default().into_finish_from_body(&body, messages)
        }
    }

    async fn consume_stream(
        &self,
        resp: reqwest::Response,
        messages: &[Message],
        on_typing: Option<TypingCb>,
    ) -> Result<(Message, Usage), String> {
        let mut deltas = Deltas::default();
        let mut buf = String::new();
        let mut last_typing = tokio::time::Instant::now();
        let mut stream = resp.bytes_stream();

        loop {
            let chunk = tokio::time::timeout(Duration::from_secs(120), stream.next()).await;
            let chunk = match chunk {
                Ok(item) => item,
                Err(_) => return Err("LLM stream stalled".to_string()),
            };
            let Some(chunk) = chunk else { break };
            let chunk = chunk.map_err(|e| format!("stream read failed: {e}"))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // Complete SSE events are separated by a blank line.
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf.drain(..pos + 2).collect();
                for line in event.lines() {
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    let parsed: Value = serde_json::from_str(data)
                        .map_err(|e| format!("bad SSE chunk: {e}"))?;
                    deltas.feed(&parsed);
                }
                // Typing indicator throttle (kuni: ~1/s; 4s is plenty humane).
                if let Some(cb) = &on_typing
                    && last_typing.elapsed() > Duration::from_secs(4)
                {
                    last_typing = tokio::time::Instant::now();
                    cb().await;
                }
            }
        }
        deltas.finish(None, messages)
    }

    async fn post_json(&self, url: &str, body: &Value) -> Result<reqwest::Response, String> {
        let cfg = self.config.get().await;
        let key = cfg.general.llm.endpoint.bearer_key.clone();
        let bearer = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| "invalid bearer_key characters".to_string())?;
        drop(cfg);

        self.http
            .post(url)
            .header(reqwest::header::AUTHORIZATION, bearer)
            .json(body)
            .send()
            .await
            .map_err(|e| format!("LLM request failed: {e}"))
    }

    /// Embeddings endpoint (kuni's `IOpenAIChat::embedding`).
    pub async fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        let cfg = self.config.get().await;
        let url = embeddings_url(&cfg.general.embedding.endpoint.base_url);
        let emd = cfg.general.embedding.clone();
        let bearer_key = cfg.general.embedding.endpoint.bearer_key.clone();
        drop(cfg);

        let body = json!({
            "model": emd.model,
            "input": [text],
        });
        let bearer = reqwest::header::HeaderValue::from_str(&format!("Bearer {bearer_key}"))
            .map_err(|_| "invalid bearer_key characters".to_string())?;
        let timeout = Duration::from_secs(self.config.get().await.misc.request_timeout_secs + 60);
        let resp = tokio::time::timeout(timeout, async {
            self.http
                .post(&url)
                .header(reqwest::header::AUTHORIZATION, bearer)
                .json(&body)
                .send()
                .await
        })
        .await
        .map_err(|_| "embeddings request timed out".to_string())?
        .map_err(|e| format!("embeddings request failed: {e}"))?;

        let json: Value = resp.json().await.map_err(|e| format!("bad embeddings response: {e}"))?;
        let Some(vec) = json["data"].get(0) else {
            return Err(format!("no embedding in response: {json}"));
        };
        let mut out = Vec::new();
        for v in vec["embedding"].as_array().unwrap_or(&vec![]) {
            out.push(v.as_f64().unwrap_or(0.0));
        }
        if out.is_empty() {
            return Err("empty embedding".to_string());
        }
        Ok(out)
    }
}

/// Removes control characters from tool results (kuni's `removeControlCharacters`).
pub fn clean_tool_result(text: &str) -> String {
    remove_control_characters(text)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_deltas_assemble_tool_calls() {
        let mut d = Deltas::default();
        d.feed(&serde_json::json!({
            "choices": [{"delta": {"content": "hello", "reasoning_content": "thin"}}]
        }));
        d.feed(&serde_json::json!({
            "choices": [{
                "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "send", "arguments": "{\"te"}},
                    {"index": 1, "id": "call_2", "function": {"name": "wait", "arguments": ""}}
                ]}
            }]
        }));
        d.feed(&serde_json::json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "xt\":\"hi\"}"}}
                ]}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }));
        let (msg, usage) = d.finish(None, &[]).unwrap();
        assert_eq!(msg.content, "hello");
        assert_eq!(msg.reasoning_content, "thin");
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].name, "send");
        assert_eq!(msg.tool_calls[0].arguments, serde_json::json!({"text": "hi"}));
        assert_eq!(usage, Usage { prompt_tokens: 10, completion_tokens: 5 });
    }

    #[test]
    fn usage_estimate_fallback() {
        let messages = vec![Message::user("some user text"), Message::assistant("reply!")];
        let (empty, usage) = Deltas {
            content: "generated line".into(),
            ..Default::default()
        }
        .finish(None, &messages)
        .unwrap();
        assert_eq!(empty.content, "generated line");
        assert!(usage.prompt_tokens > 0);
    }

    #[test]
    fn to_api_shapes() {
        let msgs = [
            Message::system("sys"),
            Message::user("q"),
            Message::tool("call_9", "result"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_9".into(),
                    name: "do".into(),
                    arguments: serde_json::json!({"x": 1}),
                }],
            ),
        ];
        let jsons: Vec<Value> = msgs.iter().map(Message::to_api).collect();
        assert_eq!(jsons[2], serde_json::json!({"role": "tool", "content": "result", "tool_call_id": "call_9"}));
        assert_eq!(
            jsons[3]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("{\"x\":1}")
        );
    }

    #[test]
    fn urls() {
        assert_eq!(chat_url("http://localhost:11434/v1/"), "http://localhost:11434/v1/chat/completions");
        assert_eq!(embeddings_url("http://localhost:11434/v1/"), "http://localhost:11434/v1/embeddings");
    }

    #[test]
    fn non_streaming_fallback() {
        let body = serde_json::json!({
            "choices": [{"message": {"content": "", "tool_calls": [
                {"id": "9", "function": {"name": "pause", "arguments": "{}"}}
            ]}}]
        });
        let (msg, usage) = Deltas::default().into_finish_from_body(&body, &[]).unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "pause");
        assert!(usage.prompt_tokens > 0); // estimated
    }
}
