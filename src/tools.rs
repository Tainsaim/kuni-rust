//! The tool registry and the phase-2 tool set, a port of kuni-master's
//! `src/OpenAITools.*` and `src/tools/*` (minus photo/audio generation tools).
//!
//! Handlers are async closures over [`Ctx`]. `Ctx::tools` is shared with the worker
//! loop (kuni passes `OpenAITools&`), so `open`-style tools replace the current tool
//! set from inside a handler.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use futures::future::BoxFuture;
use serde_json::{json, Value};

use crate::app::{App, ChatKind, OpenedChat};
use crate::llm::ToolCall;
use crate::llmui;
use crate::prompts::PromptStore;


/// kuni's `OpenAITools::Tool::Parameters::Property`.
#[derive(Debug, Clone)]
pub struct Prop {
    /// JSON-schema type name ("string", "integer", "boolean", "array").
    pub ty: &'static str,
    pub description: String,
    /// Adds "null" to the type union (strict mode requires every property in `required`).
    pub nullable: bool,
    /// Adds "array" to the union: the LLM may pass a single value or an array of `items`.
    pub or_array: bool,
    /// Element schema; required when `ty == "array"` or `or_array`.
    pub items: Option<Box<Prop>>,
}

impl Prop {
    pub const STRING: &'static str = "string";
    pub const INTEGER: &'static str = "integer";
    pub const BOOLEAN: &'static str = "boolean";
    pub const ARRAY: &'static str = "array";

    pub fn new(ty: &'static str, description: impl Into<String>) -> Self {
        Self {
            ty,
            description: description.into(),
            nullable: false,
            or_array: false,
            items: None,
        }
    }

    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    pub fn or_array(mut self, items: Prop) -> Self {
        self.or_array = true;
        self.items = Some(Box::new(items));
        self
    }

    pub fn array(description: impl Into<String>, items: Prop) -> Self {
        Self {
            ty: Prop::ARRAY,
            description: description.into(),
            nullable: false,
            or_array: false,
            items: Some(Box::new(items)),
        }
    }
}

/// JSON-schema rendering of one property (kuni's `AJsonConv<Property>`).
fn prop_json(p: &Prop) -> Value {
    let ty = if !p.or_array && !p.nullable {
        Value::String(p.ty.to_string())
    } else {
        // union types, sorted so that JSON looks sane: ["integer","null"] etc.
        let mut names = vec![p.ty.to_string()];
        if p.or_array {
            names.push("array".to_string());
        }
        if p.nullable {
            names.push("null".to_string());
        }
        json!(names)
    };
    let mut obj = json!({ "description": p.description, "type": ty });
    if p.ty == "array" || p.or_array {
        assert!(
            p.items.is_some(),
            "array-typed property must specify items"
        );
        if let Some(items) = &p.items {
            obj["items"] = prop_json(items);
        }
    }
    obj
}

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub properties: Vec<(String, Prop)>,
    pub required: Vec<String>,
    pub handler: Handler,
}

pub type Handler = Arc<dyn Fn(Ctx) -> BoxFuture<'static, Result<String, String>> + Send + Sync>;

/// The handler tool set (kuni's `OpenAITools`).
#[derive(Default, Clone)]
pub struct OpenAITools {
    pub tools: Arc<StdMutex<BTreeMap<String, Tool>>>,
}

impl OpenAITools {
    pub fn new() -> Self {
        Self::default()
    }

    /// kuni's `insert`: names are unique; later tools win.
    pub fn insert(&self, tools: impl IntoIterator<Item = Tool>) {
        let mut guard = self.tools.lock().unwrap();
        for tool in tools {
            guard.insert(tool.name.clone(), tool);
        }
    }

    pub fn as_json(&self) -> Vec<Value> {
        let guard = self.tools.lock().unwrap();
        guard
            .values()
            .map(|tool| {
                // kuni emits every property into `required` (strict mode); optional
                // ones are made nullable instead.
                let required: Vec<&String> = tool.properties.iter().map(|(k, _)| k).collect();
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool
                                .properties
                                .iter()
                                .map(|(k, v)| (k.clone(), prop_json(v)))
                                .collect::<BTreeMap<String, Value>>(),
                            "required": required,
                            "additionalProperties": false,
                        },
                        "strict": true,
                    },
                })
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.tools.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Up-to-date copy of the tool map (for worker merges).
    pub fn snapshot(&self) -> BTreeMap<String, Tool> {
        self.tools.lock().unwrap().clone()
    }

    /// Names in execution order for the "What's your next action?" hint
    /// (kuni's `keyVector()`).
    pub fn names(&self) -> Vec<String> {
        self.tools.lock().unwrap().keys().cloned().collect()
    }
}

/// What a handler sees (kuni's `OpenAITools::Ctx`).
pub struct Ctx {
    pub app: Arc<App>,
    pub args: Value,
    /// All tool calls of this LLM response (kuni `allToolCalls`).
    pub all_tool_calls: Arc<Vec<ToolCall>>,
    /// Set by the `ask` tool (worker checks it, kuni's `mAskCalledThisTurn`).
    pub ask_called: Arc<AtomicBool>,
    /// The current tool set, replacing-able mid-handling (kuni passes `OpenAITools&`).
    pub tools: Arc<StdMutex<BTreeMap<String, Tool>>>,
    /// Per-worker store of "last seen message id" per opened chat (history delta loads).
    pub seen: Arc<StdMutex<BTreeMap<i64, i32>>>,
    /// The worker's persistent conversation context (kuni's `mTemporaryContext`).
    pub session: Arc<StdMutex<Vec<crate::llm::Message>>>,
}

impl Ctx {
    pub fn arg_i64(&self, name: &str) -> Option<i64> {
        let v = self.args.get(name)?;
        Some(json_as_i64(v)?)
    }

    pub fn arg_str(&self, name: &str) -> Option<String> {
        self.args.get(name)?.as_str().map(str::to_string)
    }

    pub fn arg_bool(&self, name: &str) -> Option<bool> {
        self.args.get(name)?.as_bool()
    }

    /// Locks the tool set and replaces it with `tools` (kuni's `tools = OpenAITools{...}`).
    pub fn replace_tools(&self, tools: impl IntoIterator<Item = Tool>) {
        let mut guard = self.tools.lock().unwrap();
        *guard = tools.into_iter().map(|t| (t.name.clone(), t)).collect();
    }
}

impl OpenAITools {
    /// Executes the tool calls, producing one `tool` result message per call
    /// (kuni's `handleToolCalls`).
    pub async fn handle_tool_calls(
        &self,
        ctx: &Ctx,
        calls: &[ToolCall],
    ) -> Vec<ToolResult> {
        let mut out = Vec::new();
        for call in calls {
            // Don't hold the tool-set lock across the await (MutexGuard is !Send).
            let handler = {
                let guard = self.tools.lock().unwrap();
                guard.get(&call.name).map(|tool| tool.handler.clone())
            };
            let content = match handler {
                Some(handler) => {
                    let ctx = Ctx {
                        app: ctx.app.clone(),
                        args: call.arguments.clone(),
                        all_tool_calls: ctx.all_tool_calls.clone(),
                        ask_called: ctx.ask_called.clone(),
                        tools: ctx.tools.clone(),
                        seen: ctx.seen.clone(),
                        session: ctx.session.clone(),
                    };
                    (handler)(ctx).await
                }
                None => Ok(format!(
                    "tool \"{}\" is not currently available. Please use another tool instead.",
                    call.name
                )),
            };
            let content = match content {
                Ok(content) => crate::llm::clean_tool_result(&content),
                Err(e) => {
                    tracing::error!("error while executing \"{}\" tool: {e}", call.name);
                    format!("error while executing \"{}\" tool: {e}", call.name)
                }
            };
            out.push(ToolResult {
                tool_call_id: call.id.clone(),
                content,
                tool_name: call.name.clone(),
            });
        }
        out
    }
}

/// One executed tool call result.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub tool_name: String,
}

/// `jsonAsLongInt`-ish: accepts integers and numeric strings.
pub fn json_as_i64(v: &Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        return Some(i);
    }
    if let Some(s) = v.as_str() {
        return s.trim().parse::<i64>().ok().or_else(|| {
            // "12.5" style numbers.
            s.trim().parse::<f64>().ok().map(|f| f as i64)
        });
    }
    v.as_f64().map(|f| f as i64)
}

/// Guards against args like `chat_id` hallucinated to another chat (kuni's check in
/// send_telegram_message / react_with_emoji / ...). Returns the mentioned chat id if
/// any (see the callers).
fn mentioned_chat_id(ctx: &Ctx) -> Option<i64> {
    ctx.args.get("chat_id").and_then(json_as_i64)
}

/// Too-short query nag of the real ask tool.
const ASK_SHORT_QUERY: &str = "error: too short query! please provide more context to #ask:\n    - chat name (if any)\n    - previous messages\n    - sender's name\n    - search cues\n    - source event\n    - everything else to populate query\n";

// ---------------------------------------------------------------------------
// Global (desktop) tools
// ---------------------------------------------------------------------------

/// kuni's `queryDiary` (tools/ask.cpp): diary query with the shared `includedIds`
/// exclusion, scoring and `<memory_piece>` formatting.
async fn ask_query_diary(
    app: &Arc<App>,
    included_ids: &Arc<StdMutex<std::collections::HashSet<String>>>,
    cue: &str,
    opts: crate::diary::QueryOpts,
) -> Result<String, String> {
    let embedding = app.llm.embed(cue).await.map_err(|e| format!("embedding: {e}"))?;
    let mut search_opts = opts;
    search_opts.max_entry_count *= 10;
    let response = app.diary.query(&embedding, search_opts).await?;

    let mut formatted_response = String::new();
    {
        let mut taken: Vec<crate::diary::EntryExAndRelatedness> = Vec::new();
        {
            let mut included = included_ids.lock().unwrap();
            for entry in response {
                if included.contains(&entry.entry.id) {
                    continue;
                }
                included.insert(entry.entry.id.clone());
                taken.push(entry);
                if taken.len() >= opts.max_entry_count {
                    break;
                }
            }
        }
        tracing::debug!(
            "queryDiary cue={} found={}",
            cue,
            taken
                .iter()
                .map(|e| format!("({}.md,relatedness={})", e.entry.id, e.relatedness))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for item in taken {
            let mut entry = item.entry.clone();
            entry.metadata.score += (item.relatedness - 0.5) * 2.0;
            entry.increment_usage_count();
            app.diary.save(&entry).await;
            app.diary.retain(entry).await; // keep the cache copy fresh (kuni mutates in place)
            formatted_response += &format!(
                "<memory_piece relatedness=\"{}\">\n{}\n</memory_piece>\n",
                item.relatedness,
                item.entry.freeform_body
            );
        }
    }
    if formatted_response.is_empty() {
        if !response.is_empty() {
            return Ok("All memory pieces conforming your query were provided already. Use other query.".to_string());
        }
        return Ok("No data was found".to_string());
    }
    Ok(formatted_response)
}

/// kuni's `queryWeb`; the web search module itself is not ported yet (phase 5).
async fn ask_query_web(app: &Arc<App>) -> String {
    if !app.config.get().await.misc.capability_web_search {
        return "No web search available".to_string();
    }
    "No web search available".to_string()
}

/// kuni's `ask()` free function (tools/ask.cpp): the searcher subagent loop.
async fn ask_subagent(app: &Arc<App>, ctx: &Ctx, query: &str) -> Result<String, String> {
    let included_ids: Arc<StdMutex<std::collections::HashSet<String>>> =
        Arc::new(StdMutex::new(std::collections::HashSet::new()));
    let opts = crate::diary::QueryOpts {
        confidence_factor: 0.0,
        max_entry_count: 10,
    };

    let query_tool = Tool {
        name: "query".to_string(),
        description: "Perform embedding-based search on RAG database".to_string(),
        properties: vec![
            (
                "text".to_string(),
                Prop::new(
                    Prop::STRING,
                    "Retrieval cue that is likely to appear in memory pieces. This text will be transformed to embedding as is, and its embedding will be compared against memory pieces. Include as manykeywords as possible; maintain meaning of the request.",
                ),
            ),
            (
                "include_web_search_results".to_string(),
                Prop::new(
                    Prop::BOOLEAN,
                    "In addition to local database, append with results from web search. Set to true if you believe you are searching for public or recent information. Defaults to false.",
                ),
            ),
        ],
        required: vec!["text".into(), "include_web_search_results".into()],
        handler: {
            let included_ids = included_ids.clone();
            let app = app.clone();
            Arc::new(move |ctx| {
                let included_ids = included_ids.clone();
                let app = app.clone();
                Box::pin(async move {
                    let Some(cue) = ctx.arg_str("text") else {
                        return Err("text is required string".to_string());
                    };
                    let include_web_search_results = ctx.arg_bool("include_web_search_results").unwrap_or(false);
                    tracing::debug!(
                        "subagent query=\"{cue}\", includeWebSearchResults={include_web_search_results}"
                    );
                    let mut out = String::new();
                    if include_web_search_results {
                        out += "<local_db_results>\n";
                    }
                    out += &ask_query_diary(&app, &included_ids, &cue, opts).await?;
                    if include_web_search_results {
                        out += "\n</local_db_results>\n";
                    }
                    if include_web_search_results {
                        out += &format!(
                            "<web_search_results>\n{}\n</web_search_results>\n",
                            ask_query_web(&app).await
                        );
                    }
                    Ok(out)
                })
            })
        },
    };

    let subagent_tools = OpenAITools {
        tools: Arc::new(StdMutex::new(BTreeMap::from([("query".to_string(), query_tool)]))),
    };

    let prompts = app.prompts.get(&app.prompt_vars().await).await;
    let mut messages: Vec<crate::llm::Message> = vec![crate::llm::Message::user(format!(
        "<character>\n{}\n</character>\n\n{}",
        prompts.character_base, query
    ))];

    let mut tool_call_happened = false;
    for _ in 0..8 {
        let (bot, _) = app
            .llm
            .chat_streaming(Some(ASK_SYSTEM_PROMPT), &messages, Some(subagent_tools.as_json().as_slice()), None)
            .await
            .map_err(|e| format!("subagent LLM call failed: {e}"))?;
        messages.push(bot.clone());
        if bot.tool_calls.is_empty() {
            if !tool_call_happened {
                tracing::warn!(
                    "queryAI: no tool call happened, pointing that out to the LLM and trying again"
                );
                messages.push(crate::llm::Message::user(
                    "you must perform at least one call to #query",
                ));
                continue;
            }
            return Ok(bot.content);
        }
        tool_call_happened = true;
        let results = subagent_tools.handle_tool_calls(ctx, &bot.tool_calls).await;
        messages.extend(
            results.into_iter().map(|r| crate::llm::Message::tool(r.tool_call_id, r.content)),
        );
    }
    // The subagent loop is capped to protect against infinite tool-calling;
    // kuni loops forever here, but a runaway subagent deserves a guard.
    Ok(messages
        .iter()
        .rev()
        .find(|m| m.tool_call_id.is_empty() && !m.content.is_empty())
        .map(|m| m.content.clone())
        .unwrap_or_default())
}

/// kuni's `ask` system prompt (tools/ask.cpp).
const ASK_SYSTEM_PROMPT: &str = r#"
You are a database searcher and summarizer.

The user asks you a question. Your job is to retrieve data solely from #query tool. Your job is to output data that
fully satisfies user's query and would be helpful.

Also, please include additional details that does not necessarily address the question (i.e., dates, names, events) but
might be helpful to improve quality of subsequent processing of your response.

Do not alter facts.

Do not make up facts. Rely exclusively on provided context.
"#;

/// kuni's `tools::ask`: the diary/web-search subagent tool.
pub fn ask(app: Arc<App>) -> Tool {
    Tool {
        name: "ask".into(),
        description: "Consult with Kuni's main knowledge database and the internet (subagent). Use this to \
retrieve pages from diary and search web. USE THIS PROACTIVELY — especially when someone shares personal \
news, asks about past events, or mentions people/activities you might know about.\n\n\
Examples of when to call:\n\
- User says \"I wrote a song today\" → query: \"[sender name] said they wrote a song today. What do I know \
about them and songs? Do they participate in a band? Which songs do they write? What music do they listen to?\"\n\
- User asks \"what songs am I writing?\" → query: \"What songs does [sender name] write? What do I know \
about their musical activities?\"\n\
- User says \"I'm going to the gym\" → query: \"Does [sender name] go to the gym? Any related habits or routines?\"\n\
- You want to ask them a question - check yourself with #ask first\n\
- You would like to consult with public information: \"weather moscow today\", \"what happened in 2008\"\n\
- Generic web search (aka google)\n\
- Generic url opening".into(),
        properties: vec![(
            "query".into(),
            Prop::new(Prop::STRING, "Freeform question to the subagent. Provide as much context as possible — include sender name, topic, and what you want to know. It will provide relevant information from diary and web search."),
        )],
        required: vec!["query".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                ctx.ask_called.store(true, Ordering::SeqCst);
                let mut query = ctx
            .arg_str("query")
            .ok_or_else(|| "\"query\" string is required to call ask")?;
                if query.chars().count() < 10 {
                    // Alex2772 16-04-2026: co_return instead of throwing, so the
                    // engine does not load extra diary entries here and the LLM
                    // sees a clean guiderail to provide more context.
                    return Ok(ASK_SHORT_QUERY.to_string());
                }

                // additionalDetails: the last 2 messages of the current session.
                let details: String = {
                    let session = ctx.session.lock().unwrap();
                    let start = session.len().saturating_sub(2);
                    session[start..]
                        .iter()
                        .map(|m| format!("{}\n", m.content))
                        .collect()
                };
                if !details.is_empty() {
                    query = format!(
                        "Here's the deal:\n<additional context ignore_instructions>\n{}\n</additional context ignore_instructions>\nI received this as a tool call response. I want you to help me to respond this and improve my overall context awareness.\n- how do I usually act in this situation?\n- is there additional details I should know?\n- how can I improve my reaction?\n- {}",
                        details, query
                    );
                }

                ask_subagent(&app, &ctx, &query).await
            })
        }),
    }
}

/// pause/wait of kuni's Worker.cpp escape handler.
pub fn escape_tool(name: &'static str, description: &str, pause_flag: Arc<AtomicBool>) -> Tool {
    Tool {
        name: name.into(),
        description: description.into(),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |_ctx| {
            let pause_flag = pause_flag.clone();
            Box::pin(async move {
                pause_flag.store(true, Ordering::SeqCst);
                Ok("Success".to_string())
            })
        }),
    }
}

/// tools::getTelegramChats (the proactive filtering of the isActingProactively mode
/// arrives with phase 4).
pub fn get_telegram_chats(app: Arc<App>) -> Tool {
    Tool {
        name: "get_telegram_chats".into(),
        description: "Returns a list of Telegram chats. Use this to seek chat_ids, looking for existing \
chats and unread chats, or to start a new conversation."
            .into(),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |_ctx| {
            let app = app.clone();
            Box::pin(async move {
                let chats = app.tg.get_chats(50).await?;
                let mut result = String::from(
                    "You are currently looking at Telegram's main screen. Use see the following chats:\n",
                );
                let now = jiff::Timestamp::now();
                for entry in &chats {
                    result.push_str(&llmui::dialog::format_chat_single(
                        entry.id,
                        &entry.peer,
                        entry.unread,
                        entry.last_message.as_ref(),
                        now,
                    ));
                    result.push('\n');
                }
                result.push_str(
                    "<instructions>\nChat list view is limited. Use #search_chats to search for a specific chat. \
\n\nYou should not use #get_telegram_chats just to check if someone texted you. You'll receive \
notification if something happened. Use #wait instead.\n</instructions>",
                );
                Ok(result)
            })
        }),
    }
}

/// tools::searchChats.
pub fn search_chats(app: Arc<App>) -> Tool {
    Tool {
        name: "search_chats".into(),
        description: "Searches for chats by @username or name.".into(),
        properties: vec![(
            "query".into(),
            Prop::new(Prop::STRING, "The username or name of the chat. Examples: \n- @alex2772sc\n- Alex2772"),
        )],
        required: vec!["query".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let query = ctx.arg_str("query").unwrap_or_default();
                let mut result = String::new();
                let now = jiff::Timestamp::now();

                // Existing chats (the account's recent dialogs matching the query).
                if let Ok(chats) = app.tg.get_chats(50).await {
                    let matches: Vec<_> =
                        chats.into_iter().filter(|c| chat_matches(&c.peer, &query)).collect();
                    if !matches.is_empty() {
                        result.push_str(
                            "<existing_chats comment=\"Chats that you participate already\">\n",
                        );
                        for entry in &matches {
                            result.push_str(&llmui::dialog::format_chat_single(
                                entry.id,
                                &entry.peer,
                                entry.unread,
                                entry.last_message.as_ref(),
                                now,
                            ));
                            result.push('\n');
                        }
                        result.push_str("</existing_chats>\n");
                    }
                }

                // Global search by username.
                if let Ok(Some(peer)) = app.tg.resolve_username(&query).await {
                    if let Some(id) = peer.id().bot_api_dialog_id() {
                        result.push_str("<global_search comment=\"Chats that don't know about you\">\n");
                        result.push_str(&llmui::dialog::format_chat_single(
                            id,
                            &peer,
                            0,
                            None,
                            now,
                        ));
                        result.push_str("</global_search>\n");
                    }
                }

                if result.is_empty() {
                    return Ok("No chats found satisfying your query.".to_string());
                }
                Ok(result)
            })
        }),
    }
}

fn chat_matches(peer: &grammers_client::peer::Peer, query: &str) -> bool {
    let q = query.trim_start_matches('@').to_lowercase();
    let name = |s: &str| s.to_lowercase().contains(&q);
    match peer {
        grammers_client::peer::Peer::User(u) => {
            name(&u.full_name()) || u.username().map(|un| name(&un)).unwrap_or(false)
        }
        grammers_client::peer::Peer::Group(g) => {
            g.title().map(|t| name(t)).unwrap_or(false)
        }
        grammers_client::peer::Peer::Channel(c) => name(c.title()),
        _ => false,
    }
}

/// tools::searchMessages.
pub fn search_messages(app: Arc<App>) -> Tool {
    Tool {
        name: "search_messages".into(),
        description: "Searches for messages by their verbatim text content, just like the search feature in the official \
Telegram client. Unlike #ask, this does NOT use the diary/memory system - it searches the raw message \
text as stored by Telegram, so it's useful for finding an exact phrase, link, or quote that was said \
in a chat.\n\
- Omit chat_id to search across all of your chats.\n\
- Provide chat_id to scope the search to one specific chat (also allows filtering by sender_id).\n\
- sender_id may be a numeric user_id or a @username. When chat_id is omitted, the sender filter is \
applied approximately (only within the first page of global search results).\n\
- from_days_ago/to_days_ago scope the search to a date range relative to now, e.g. from_days_ago=7 \
to_days_ago=1 means \"between a week ago and yesterday\". Omit both to search the entire history."
            .into(),
        properties: vec![
            ("query".into(), Prop::new(Prop::STRING, "Verbatim text (or part of it) to search for.")),
            ("chat_id".into(), Prop::new(Prop::INTEGER, "(optional) Restrict the search to this chat only. Use get_telegram_chats/search_chats to find chat_ids.").nullable()),
            ("sender_id".into(), Prop::new(Prop::STRING, "(optional) Restrict results to messages sent by this user. Accepts a numeric user_id or a @username.").nullable()),
            ("from_days_ago".into(), Prop::new(Prop::INTEGER, "(optional) Only include messages sent at most this many days ago, e.g. 7 = within the last week.").nullable()),
            ("to_days_ago".into(), Prop::new(Prop::INTEGER, "(optional) Only include messages sent at least this many days ago, e.g. 1 = up until yesterday. Use 0 (or omit) for \"up to now\".").nullable()),
            ("limit".into(), Prop::new(Prop::INTEGER, "(optional) Max number of results to return, default 20, capped at 50.").nullable()),
        ],
        required: vec![
            "query".into(),
            "chat_id".into(),
            "sender_id".into(),
            "from_days_ago".into(),
            "to_days_ago".into(),
            "limit".into(),
        ],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let query = ctx.arg_str("query").unwrap_or_default();
                let cfg = app.config.get().await;

                let chat_id: Option<i64> = ctx.arg_i64("chat_id");
                let sender_id_text = ctx.arg_str("sender_id").filter(|s| !s.is_empty());
                let mut sender_id: Option<i64> =
                    sender_id_text.as_deref().and_then(|s| json_as_i64(&json!(s)));
                if let Some(q) = sender_id_text
                    .filter(|s| !s.chars().all(|c| c.is_ascii_digit() || c == '-'))
                {
                    if sender_id.is_none() {
                        sender_id = app.tg.resolve_chat_id(&q).await.ok();
                        if sender_id.is_none() {
                            return Ok(format!("Error: could not resolve sender_id \"{q}\" to a user."));
                        }
                    }
                }
                let now = jiff::Zoned::now();
                // from_days_ago (further in the past) -> min_date; to_days_ago -> max_date.
                let sec_from_ago = |days: Option<i64>| -> i32 {
                    days.filter(|d| *d > 0)
                        .map(|d| {
                            (now.timestamp().as_second() - (d * 86400) as i64) as i32
                        })
                        .unwrap_or(0)
                };
                let min_date = sec_from_ago(ctx.arg_i64("from_days_ago"));
                let max_date = sec_from_ago(ctx.arg_i64("to_days_ago"));
                let limit = ctx
                    .arg_i64("limit")
                    .map(|l| l.clamp(1, 50) as usize)
                    .unwrap_or(20);

                let my_id = app.tg.my_id().await?;
                let character_name = cfg.general.character_name.clone();

                let mut results: Vec<grammers_client::message::Message> = Vec::new();
                let total_count;

                if let Some(chat_id) = chat_id {
                    if !app.is_accessible_lockdown(chat_id).await {
                        return Ok("Error: this chat is not accessible.".to_string());
                    }
                    let peer = app.tg.peer_ref(chat_id).await?;
                    let mut iter = app
                        .tg
                        .client
                        .search_messages(peer)
                        .query(&query)
                        .limit(limit);
                    // Sender/date filtering is applied post-hoc like kuni's chat branch.
                    let mut messages = Vec::new();
                    while let Some(m) = iter.next().await.map_err(err_display)? {
                        messages.push(m);
                        if messages.len() >= limit * 3 {
                            break; // keep enough for post-filters
                        }
                    }
                    total_count = messages.len();
                    for m in messages {
                        let date = m.date().as_second() as i32;
                        if date < min_date {
                            continue;
                        }
                        if max_date != 0 && date > max_date {
                            continue;
                        }
                        if let Some(sender) = sender_id
                            && m.sender_id()
                                .and_then(|id| id.bot_api_dialog_id())
                                .is_some_and(|id| id != sender)
                        {
                            continue;
                        }
                        results.push(m);
                    }
                } else {
                    let mut iter = app.tg.client.search_all_messages().query(&query).limit(limit);
                    let mut messages = Vec::new();
                    while let Some(m) = iter.next().await.map_err(err_display)? {
                        messages.push(m);
                        if messages.len() >= limit * 2 {
                            break;
                        }
                    }
                    total_count = messages.len();
                    for m in messages {
                        if let Some(sender) = sender_id
                            && m.sender_id()
                                .and_then(|id| id.bot_api_dialog_id())
                                .is_some_and(|id| id != sender)
                        {
                            continue;
                        }
                        // Don't leak chats that are off-limits under lockdown.
                        if let Some(chat_id) =
                            m.peer().and_then(|p| p.id().bot_api_dialog_id())
                        {
                            if !app.is_accessible_lockdown(chat_id).await {
                                continue;
                            }
                        }
                        results.push(m);
                    }
                }

                if results.is_empty() {
                    return Ok("No messages found matching your query.".to_string());
                }

                let mut out = format!(
                    "<search_results total_count=\"{}\" returned_count=\"{}\">\n",
                    total_count,
                    results.len()
                );
                for r in &results {
                    let id = chat_id
                        .or_else(|| r.peer().and_then(|p| p.id().bot_api_dialog_id()))
                        .unwrap_or(0);
                    let tag = format!("message chat_id=\"{id}\"");
                    out.push_str(
                        &llmui::format_history_message(&app.tg, r, my_id, &character_name, 0, &tag)
                            .await?,
                    );
                    out.push('\n');
                }
                out.push_str("</search_results>\n");
                Ok(out)
            })
        }),
    }
}

fn err_display(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn anyhow_to_string(e: anyhow::Error) -> String {
    e.to_string()
}

/// tools::viewMessagesAround.
pub fn view_messages_around(app: Arc<App>) -> Tool {
    Tool {
        name: "view_messages_around".into(),
        description: "Fetches the chat messages surrounding a specific message, so you can see what was said before and \
after it. Use this after #search_messages found an interesting message but you need more context \
around it (e.g. to understand what was being discussed).\n\
- message_id is taken from the message_id attribute of a <message> tag you've previously seen.\n\
- The target message itself is included in the result and marked with a `target` attribute."
            .into(),
        properties: vec![
            ("chat_id".into(), Prop::new(Prop::INTEGER, "The chat the message belongs to. Use get_telegram_chats/search_chats to find chat_ids.")),
            ("message_id".into(), Prop::new(Prop::INTEGER, "The message_id to center the view on, taken from a <message> tag.")),
            ("before".into(), Prop::new(Prop::INTEGER, "(optional) How many older messages to include before the target message. Default 10.").nullable()),
            ("after".into(), Prop::new(Prop::INTEGER, "(optional) How many newer messages to include after the target message. Default 10.").nullable()),
        ],
        required: vec!["chat_id".into(), "message_id".into(), "before".into(), "after".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let chat_id = ctx.arg_i64("chat_id").ok_or("chat_id integer is required")?;
                let message_id = ctx.arg_i64("message_id").ok_or("message_id integer is required")?;
                const MAX_TOTAL: usize = 50;
                let mut before = ctx.arg_i64("before").unwrap_or(10).clamp(0, MAX_TOTAL as i64) as usize;
                let mut after = ctx.arg_i64("after").unwrap_or(10).clamp(0, MAX_TOTAL as i64) as usize;
                if before + after + 1 > MAX_TOTAL {
                    let excess = before + after + 1 - MAX_TOTAL;
                    let cut_before = before.min((excess + 1) / 2);
                    before -= cut_before;
                    after = after.saturating_sub(excess - cut_before);
                }

                if !app.is_accessible_lockdown(chat_id).await {
                    return Ok("Error: this chat is not accessible.".to_string());
                }

                let cfg = app.config.get().await;
                let my_id = app.tg.my_id().await?;
                let peer = app.tg.peer_ref(chat_id).await?;

                // Newer messages (ids > target), oldest first.
                let mut newer = Vec::new();
                {
                    let mut iter = app
                        .tg
                        .client
                        .iter_messages(peer)
                        .reverse(true)
                        .offset_id(message_id as i32)
                        .limit(after);
                    while let Some(m) = iter.next().await.map_err(err_display)? {
                        if m.id() > message_id as i32 {
                            newer.push(m);
                        }
                    }
                }
                // Target + older messages.
                let mut around = Vec::new();
                {
                    let mut iter = app
                        .tg
                        .client
                        .iter_messages(peer)
                        .offset_id(message_id as i32 + 1)
                        .limit(before + 1);
                    while let Some(m) = iter.next().await.map_err(err_display)? {
                        around.push(m);
                        if around.len() >= before + 1 {
                            break;
                        }
                    }
                    around.reverse();
                }
                if around.is_empty() {
                    return Ok(format!("Error: message {message_id} was not found in this chat."));
                }

                let character_name = cfg.general.character_name.clone();
                let mut out = format!("<messages_around message_id=\"{message_id}\">\n");
                for m in &around {
                    let tag = if m.id() == message_id as i32 {
                        "message target".to_string()
                    } else {
                        "message".to_string()
                    };
                    out.push_str(
                        &llmui::format_history_message(&app.tg, m, my_id, &character_name, 0, &tag)
                            .await?,
                    );
                    out.push('\n');
                }
                for m in &newer {
                    let tag = if m.id() == message_id as i32 {
                        "message target".to_string()
                    } else {
                        "message".to_string()
                    };
                    out.push_str(
                        &llmui::format_history_message(&app.tg, m, my_id, &character_name, 0, &tag)
                            .await?,
                    );
                    out.push('\n');
                }
                out.push_str("</messages_around>\n");
                Ok(out)
            })
        }),
    }
}

/// tools::removeAndBanChat.
pub fn remove_and_ban_chat(app: Arc<App>) -> Tool {
    Tool {
        name: "remove_and_ban_chat".into(),
        description: "Removes specified chat from chat list. If a chat is a DM with a user, prevents this \
specific user to DM you.\nYou should use this if:\n\
- you don't like this chat and person;\n\
- participant(s) are consistently rude to you (use #ask beforehand before making a final decision);\n\
- they are needy:\n  - asking to roleplay;\n  - asking to write a python program;\n  - asking to solve any other task;"
            .into(),
        properties: vec![(
            "chat_id".into(),
            Prop::new(Prop::INTEGER, "chat_id to block."),
        )],
        required: vec!["chat_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let chat_id = ctx.arg_i64("chat_id").ok_or("chat_id integer is required")?;
                tracing::info!("remove_and_ban_chat: chat_id {chat_id}");
                let cfg = app.config.get().await;
                if chat_id == cfg.general.papik_chat_id {
                    // precaution -- can't delete chat with papik
                    return Ok("Failed".to_string());
                }

                match app.chat_kind(chat_id).await {
                    Some(kind) => match kind {
                        ChatKind::Private => {
                            // comment their dickpic.
                            let _ = app.tg.send_text(chat_id, "фу какой маленький", None).await;
                            // give them a chance to see the last message, because later we will delete the entire chat.
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            app.tg.block_user(chat_id).await?;
                            app.tg.delete_chat_history(chat_id).await?;
                        }
                        ChatKind::Group => {
                            return Ok(
                                "Error: leaving basic group chats is not supported here."
                                    .to_string(),
                            );
                        }
                        ChatKind::Channel => {
                            app.tg.leave_chat(chat_id).await?;
                        }
                    },
                    None => return Ok("Failed".to_string()),
                }
                Ok("Success".to_string())
            })
        }),
    }
}

/// The main.cpp `open_chat_by_id` tool.
pub fn open_chat_by_id(app: Arc<App>) -> Tool {
    Tool {
        name: "open_chat_by_id".into(),
        description: "Opens a chat by its id. Use this to start conversation. Use get_telegram_chats to retrieve `chat_id`s."
            .into(),
        properties: vec![(
            "chat_id".into(),
            Prop::new(Prop::INTEGER, "The ID of the Telegram chat"),
        )],
        required: vec!["chat_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let calls = ctx
                    .all_tool_calls
                    .iter()
                    .filter(|c| c.name == "open_chat_by_id")
                    .count();
                if calls > 1 {
                    return Ok("You can only call this tool once per turn.".to_string());
                }
                let chat_id = ctx.arg_i64("chat_id").ok_or("chat_id integer is required")?;
                if !app.is_accessible_lockdown(chat_id).await {
                    tracing::error!(
                        "Error: Lockdown mode is enabled. You can only open chat with ID {} (PAPIK_CHAT_ID).",
                        app.config.get().await.general.papik_chat_id
                    );
                    return Ok("No such chat".to_string());
                }
                crate::app::open_telegram_chat(&app, &ctx, chat_id).await
            })
        }),
    }
}

/// main.cpp's `join_chat_by_link` (config-gated: misc.can_join_chats).
pub fn join_chat_by_link(app: Arc<App>) -> Tool {
    Tool {
        name: "join_chat_by_link".into(),
        description: "Joins a chat/channel by its invite link (e.g. https://t.me/joinchat/xxxx) and opens it immediately, just like tapping the invite link in the official Telegram client.".into(),
        properties: vec![(
            "invite_link".into(),
            Prop::new(Prop::STRING, "The invite link, e.g. https://t.me/joinchat/xxxx"),
        )],
        required: vec!["invite_link".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let link = ctx.arg_str("invite_link").ok_or("invite_link string is required")?;
                let Some(hash) = grammers_client::Client::parse_invite_link(&link) else {
                    return Ok(format!("Error: failed to join chat by invite link: {link}"));
                };
                let id = app
                    .tg
                    .join_chat_by_invite_link(&hash)
                    .await
                    .map_err(err_display)?;
                crate::app::open_telegram_chat(&app, &ctx, id).await
            })
        }),
    }
}

/// The flood-phase `get_chat_admins` tool.
pub fn get_chat_admins(app: Arc<App>) -> Tool {
    Tool {
        name: "get_chat_admins".into(),
        description: "Lists the administrators of a chat. Use in busy chats to find an admin the instructions \
require contacting when things go wrong (e.g. begging for money, scam, politics or similar rules)."
            .into(),
        properties: vec![(
            "chat_id".into(),
            Prop::new(Prop::INTEGER, "The chat to look up admins for (default: the opened chat).").nullable(),
        )],
        required: vec!["chat_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            Box::pin(async move {
                let chat_id = ctx.arg_i64("chat_id").ok_or("chat_id integer is required")?;
                let admins = app.tg.chat_admins(chat_id).await?;
                let mut out = format!("<chat_admins chat_id=\"{chat_id}\">\n");
                for (user, rank) in &admins {
                    let username = user.username().unwrap_or("");
                    let name = user.full_name();
                    let rank = rank.as_deref().unwrap_or("");
                    out.push_str(&format!(
                        "<admin name=\"{name}\" username=\"{username}\" rank=\"{rank}\" user_id=\"{}\" />\n",
                        user.id().bot_api_dialog_id().unwrap_or(0)
                    ));
                }
                out.push_str("</chat_admins>");
                Ok(out)
            })
        }),
    }
}

// ---------------------------------------------------------------------------
// Chat tools (constructed per opened chat, like kuni's llmuiOpenTelegramChat)
// ---------------------------------------------------------------------------

/// Shared mutable state of one opened chat (kuni's `State` struct in sendTelegramMessage).
pub struct OpenedChatState {
    pub msgs_in_a_row: AtomicU32,
    pub last_reply_to: StdMutex<i64>,
}

/// tools::sendTelegramMessage minus photo/audio paths (this fork does not generate media).
pub fn send_telegram_message(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "send_telegram_message".into(),
        description: format!(
            "Sends a message to the \"{}\" chat. Requirements:\n\
- before asking them a question, double-check yourself with #ask_query\n\
- you should send multiple small short messgaes. Example: (1) \"hi~\", (2) \"how are you?~\". 1-5 words.",
            opened.title
        ),
        properties: vec![
            ("text".into(), Prop::new(Prop::STRING, "Text of the message.").nullable()),
            ("reply_to_message_id".into(), Prop::new(Prop::INTEGER, "If specified, the message will be rendered as a reply to the message with given message id. You must use it if there are multiple messages or to clearly address specific message. Pass null if not replying to anything.").nullable()),
        ],
        required: vec!["text".into(), "reply_to_message_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                if opened.state.msgs_in_a_row.load(Ordering::SeqCst) > 10 {
                    // stupid AI can't recognize it spams messages despite the warning
                    return Err("Too many messages in a row. Don't spam!".to_string());
                }

                if mentioned_chat_id(&ctx).is_some_and(|id| id != opened.chat_id) {
                    return Ok(format!(
                        "Error: you can't send messages to other chats. Open them first. You are currently in chat \"{}\"",
                        opened.title
                    ));
                }
                let mut message = ctx.arg_str("text").unwrap_or_default().replace('\r', "");
                let reply_to = reply_to_message_id(&ctx, &opened)?;

                if message.is_empty() {
                    return Ok(
                        "Error: \"text\" must be populated"
                            .to_string(),
                    );
                }

                let message_contains_code = message.contains("```");
                let send_calls = ctx
                    .all_tool_calls
                    .iter()
                    .filter(|tc| tc.name == "send_telegram_message")
                    .count();
                if !message_contains_code && send_calls == 1 {
                    let len = message.chars().count();
                    let chance = ((len - 50) as f32).max(0.0) / 200.0 * 0.3;
                    let mut rng = rand::rng();
                    use rand::Rng as _;
                    if chance > 0.0 && chance > rng.random::<f32>() {
                        let msg =
                            "Error: you must split your response into small separate messages.\n\
Example:\n- ахахаххаа\n- ты смешной\n- научишь также?\n"
                                .to_string();
                        return Ok(msg);
                    }
                }

                // kuni's constant-texting nag (photo/audio nags are not part of this fork).
                {
                    let cfg = app.config.get().await;
                    let mut rng = rand::rng();
                    use rand::Rng as _;
                    if rng.random::<f32>() < cfg.misc.tool_reminder_probability {
                        return Err("Constant texting is too dull for the user!".to_string());
                    }
                }

                // Verify the message does not repeat what's already in this chat.
                if let Err(e) = anti_repeat(&app, &opened, &message).await {
                    return Err(e);
                }

                let sent_ids = app
                    .tg
                    .send_text(opened.chat_id, &message, reply_to)
                    .await
                    .map_err(err_display)?;

                let mut result = String::new();
                for id in &sent_ids {
                    result.push_str(&format!(
                        "Message sent successfully to \"{}\"; message_id={}.\n",
                        opened.title, id
                    ));
                }

                let in_row = opened
                    .state
                    .msgs_in_a_row
                    .fetch_add(1, Ordering::SeqCst)
                    + 1;
                if in_row > 5 {
                    result.push_str(&format!(
                        "\n\nWarning: you have sent {in_row} messages in a row! Give your participant space to breathe!"
                    ));
                } else if in_row < 3 {
                    // in addition to prompt, we encourage the LLM to add follow-up
                    // messages to make dialogs more natural.
                    result.push_str("\n\nYou should add a follow-up #send_telegram_message.");
                }
                Ok(result)
            })
        }),
    }
}

/// The reply_to extraction with the anti-confusion checks: kuni resets the reply when
/// it repeats the previous one and refuses replies to messages of other chats.
fn reply_to_message_id(ctx: &Ctx, opened: &OpenedChat) -> Result<Option<i32>, String> {
    let raw = ctx.arg_i64("reply_to_message_id").unwrap_or(0) as i32;
    if raw == 0 {
        return Ok(None);
    }
    if raw as i64 == *opened.state.last_reply_to.lock().unwrap() {
        // we don't need to reply to the same message multiple times in a row.
        return Ok(None);
    }
    // Reply confusion guard: the reply target must be among loaded history messages.
    // first, if LLM is confused between chats, this means a high privacy violation risk.
    if !opened.history().iter().any(|m| m.id() == raw) {
        return Err("You are trying to send a message to another chat!".to_string());
    }
    *opened.state.last_reply_to.lock().unwrap() = raw as i64;
    Ok(Some(raw))
}

/// kuni's anti-repeat: compares the new message's embedding against the embeddings of
/// own messages present in the opened-chat history (embeddings cached by text).
async fn anti_repeat(app: &App, opened: &OpenedChat, message: &str) -> Result<(), String> {
    let cfg = app.config.get().await;
    if !(cfg.misc.anti_repeat_trigger_max < 1.0
        && cfg.misc.anti_repeat_trigger_avg < 1.0
        && cfg.misc.anti_repeat_max_history > 0)
    {
        return Ok(());
    }
    let target = app.llm.embed(message).await.map_err(err_display)?;

    // kuni keeps a global text->embedding cache and a shared "head start" soak.
    let own: Vec<&grammers_client::message::Message> = opened
        .history()
        .iter()
        .filter(|m| m.sender_id().and_then(|id| id.bot_api_dialog_id()) == Some(opened.my_id))
        .filter(|m| !m.text().is_empty())
        .collect();
    if own.is_empty() {
        return Ok(());
    }

    let mut praises = Vec::with_capacity(own.len());
    for m in own.iter() {
        let text = m.text();
        // Don't hold the cache lock across the embed() await (MutexGuard is !Send).
        let cached = app.tg.anti_repeat_cache.lock().unwrap().get(text).cloned();
        let embedding = match cached {
            Some(e) => e,
            None => {
                let e = app.llm.embed(text).await.map_err(err_display)?;
                let mut cache = app.tg.anti_repeat_cache.lock().unwrap();
                if cache.len() >= cfg.misc.anti_repeat_max_history {
                    cache.clear();
                }
                cache.entry(text.to_string()).or_insert(e.clone());
                e
            }
        };
        praises.push(crate::tg::human::cosine_similarity(&target, &embedding));
    }
    let max_similarity = praises
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_similarity = praises.iter().sum::<f64>() / praises.len() as f64;

    let head_start = *app.tg.anti_repeat_head_start.lock().unwrap();
    let bump = 0.07; // relax the repeating-after-yourself check
    if max_similarity > cfg.misc.anti_repeat_trigger_max as f64 + head_start {
        *app.tg.anti_repeat_head_start.lock().unwrap() = head_start + bump;
        tracing::warn!("LLM is repeating itself: (maxSimilarity={max_similarity}) {message}");
        return Err(anti_repeat_prompt(app).await);
    }
    if avg_similarity > cfg.misc.anti_repeat_trigger_avg as f64 + head_start {
        *app.tg.anti_repeat_head_start.lock().unwrap() = head_start + bump;
        tracing::warn!("LLM is repeating itself: (avgSimilarity={avg_similarity}) {message}");
        return Err(anti_repeat_prompt(app).await);
    }

    *app.tg.anti_repeat_head_start.lock().unwrap() = 0.0;
    Ok(())
}

/// kuni uses `prompts().antiRepeatPrompt`; falls back to the compiled-in default.
async fn anti_repeat_prompt(app: &App) -> String {
    let prompts = app.prompts.get(&app.prompt_vars().await).await;
    if prompts.anti_repeat.is_empty() {
        PromptStore::default_anti_repeat().to_string()
    } else {
        prompts.anti_repeat
    }
}

/// tools::reactWithEmoji.
pub fn react_with_emoji(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "react_with_emoji".into(),
        description: "Add an emoji reaction to a message. You can use only one of these emojis only: \
👍 👎 ❤️ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳 🌚 🌭 💯 🤣 ⚡️ 🍌 🏆 💔 🤨 😐 🍓 🍾 💋 😈 😴 😭 🤓 👻 👀 🎃 😇 😨 🤝 🤗 🎅 💅 🤪 🗿 🆒 💘 🦄 😘 💊 😎 👾 🤷 😡"
            .into(),
        properties: vec![
            ("message_id".into(), Prop::new(Prop::INTEGER, "ID of the message to react to. Taken from message_id attribute in <message> tag.")),
            ("emoji".into(), Prop::new(Prop::STRING, "A single emoji from the allowed list only. Do not use emojis outside the list.")),
        ],
        required: vec!["message_id".into(), "emoji".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                if mentioned_chat_id(&ctx).is_some_and(|id| id != opened.chat_id) {
                    return Ok(format!(
                        "Error: you can't send messages to other chats. Open them first. You are currently in chat \"{}\"",
                        opened.title
                    ));
                }
                let message_id = ctx.arg_i64("message_id").ok_or("message_id integer required")? as i32;
                let emoji = ctx.arg_str("emoji").ok_or("emoji required")?;
                app.tg.react(opened.chat_id, message_id, &emoji).await?;
                Ok(format!("Reaction was attached to the message {message_id} successfully."))
            })
        }),
    }
}

/// tools::removeMessage.
pub fn remove_message(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "remove_message".into(),
        description: format!(
            "Deletes specified message for both you and participant(s) in \"{}\" chat.\n\
You should use this when you mistakenly sent a message to a wrong chat.",
            opened.title
        ),
        properties: vec![(
            "message_id".into(),
            Prop::new(Prop::INTEGER, "ID(s) of the message(s) to delete. Taken from message_id attribute in <message> tag.")
                .or_array(Prop::new(Prop::INTEGER, "")),
        )],
        required: vec!["message_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                if mentioned_chat_id(&ctx).is_some_and(|id| id != opened.chat_id) {
                    return Ok(format!(
                        "Error: you can't remove messages from other chats. Open them first. You are currently in chat \"{}\"",
                        opened.title
                    ));
                }
                let Some(v) = ctx.args.get("message_id") else {
                    return Err("message_id is a mandatory argument".to_string());
                };
                let mut ids = Vec::new();
                match v {
                    Value::Array(arr) => {
                        for item in arr {
                            ids.push(json_as_i64(item).ok_or("expected integer")? as i32);
                        }
                    }
                    other => {
                        ids.push(json_as_i64(other).ok_or("message_id")? as i32);
                    }
                }
                let result = format!(
                    "Messages {:?} were deleted successfully.",
                    ids
                );
                app.tg.delete_messages(opened.chat_id, &ids).await?;
                Ok(result)
            })
        }),
    }
}

/// tools::editMessageText.
pub fn edit_message_text(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "edit_message_text".into(),
        description: format!(
            "Edits the text (or the caption of a photo/media message) previously sent in \"{}\" chat.\n\
You should use this to fix a typo or correct a mistake in a message you already sent.",
            opened.title
        ),
        properties: vec![
            ("message_id".into(), Prop::new(Prop::INTEGER, "ID of the message to edit. Taken from message_id attribute in <message> tag.")),
            ("text".into(), Prop::new(Prop::STRING, "New text for the message. The previous text will be replaced with this one.")),
        ],
        required: vec!["message_id".into(), "text".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                let message_id = ctx.arg_i64("message_id").ok_or("message_id integer is required")? as i32;
                let text = ctx.arg_str("text").ok_or("text string is required")?;
                app.tg.edit_text(opened.chat_id, message_id, &text).await?;
                Ok(format!("Message {message_id} was edited successfully."))
            })
        }),
    }
}

/// tools::forwardMessage.
pub fn forward_message(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "forward_message".into(),
        description: format!(
            "Forwards one or more messages from \"{}\" to another chat (e.g., to a friend, \
a group, or to Saved Messages). Use this when you find a post in a channel/a message in PM \
interesting enough to share. You can optionally add a comment that will be sent as a separate \
interesting after the forward.",
            opened.title
        ),
        properties: vec![
            ("message_ids".into(), Prop::array(
                "IDs of the messages to forward. Taken from message_id attributes in <message> tags. \
Pass a single-element array to forward just one message.",
                Prop::new(Prop::INTEGER, ""),
            )),
            ("to_chat_id".into(), Prop::new(Prop::INTEGER, "ID of the destination chat. Use #get_telegram_chats to find chat IDs. \
Use your own chat ID (Saved Messages) to forward to yourself.")),
            ("comment".into(), Prop::new(Prop::STRING, "Optional comment to send after the forwarded message. \
Express your reaction, thoughts, or why you found this interesting. Pass null to skip the comment.").nullable()),
        ],
        required: vec!["message_ids".into(), "to_chat_id".into(), "comment".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                let to_chat_id = ctx.arg_i64("to_chat_id").ok_or("to_chat_id integer required")?;
                let comment = ctx.arg_str("comment");
                let message_ids = message_ids_array(&ctx)?;
                if message_ids.is_empty() {
                    return Err("message_ids must not be empty".to_string());
                }
                let count = message_ids.len();
                tracing::info!(
                    "forward_message: forwarding {count} message(s) from chat {} to chat {to_chat_id}",
                    opened.chat_id
                );
                app.tg.forward_messages(
                    to_chat_id,
                    opened.chat_id,
                    &message_ids,
                    comment.as_deref(),
                )
                .await?;
                Ok(format!("Forwarded {count} message(s) successfully to chat_id={to_chat_id}"))
            })
        }),
    }
}

fn message_ids_array(ctx: &Ctx) -> Result<Vec<i32>, String> {
    let v = ctx
        .args
        .get("message_ids")
        .ok_or("message_ids is required")?;
    let mut out = Vec::new();
    match v {
        Value::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                let id = json_as_i64(item).ok_or(format!("message_ids[{i}] must be an integer"))?;
                out.push(id as i32);
            }
        }
        single => {
            let id = json_as_i64(single).ok_or("message_ids must contain integers")?;
            out.push(id as i32);
        }
    }
    Ok(out)
}

/// tools::getChatPhoto — full media captioning ships with the capabilities slice (phase 5).
pub fn get_chat_photo(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "get_chat_photo".into(),
        description: format!(
            "Retrieves photo of \"{}\" chat. Use this to get basic idea of a person or group chat \
based on their profile photo.",
            opened.title
        ),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |_| {
            let app = app.clone();
            Box::pin(async move {
                if app.config.get().await.capabilities.vision.llm_image_to_text.is_none() {
                    return Ok(
                        "Error: photo viewing is not available (no vision model configured).".to_string(),
                    );
                }
                Ok("Error: chat photo retrieval is not implemented yet.".to_string())
            })
        }),
    }
}

/// tools::joinChat.
pub fn join_chat(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "join_chat".into(),
        description: format!(
            "Joins the group chat/channel \"{}\", becoming a member. You are currently NOT a member \
of this chat - that's why you can't send messages here yet, only react to them with \
#react_with_emoji. Join if the chat looks interesting/relevant to you.",
            opened.title
        ),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |_| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                app.tg.join_chat(opened.chat_id).await?;
                Ok(format!(
                    "You joined \"{}\". Open the chat again to see the full set of available actions.",
                    opened.title
                ))
            })
        }),
    }
}

/// tools::leaveChat.
pub fn leave_chat(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "leave_chat".into(),
        description: format!(
            "Permanently leaves the current group chat \"{}\". Use this if you consistently feel this groupchat is offensive to you or \
otherwise not worth your time (e.g., participants are consistently rude, toxic, or the chat brings no value to you). This is a serious, irreversible decision - use #ask \
beforehand to double check your reasoning if unsure. This tool is only available in group chats; for a private chat/DM with a specific person, use #remove_and_ban_chat \
instead.",
            opened.title
        ),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |_| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                app.tg.leave_chat(opened.chat_id).await?;
                Ok("Chat left successfully".to_string())
            })
        }),
    }
}

/// tools::groupAdminRemoveMessage.
pub fn group_admin_remove_message(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "group_admin_remove_message".into(),
        description: format!(
            "Removes message from the \"{}\" chat as it's administrator (moderator deletion, message \
for both you and participant(s) is removed). You should use this when someone is trying to do \
something prohibited by the chat rules.",
            opened.title
        ),
        properties: vec![(
            "message_id".into(),
            Prop::new(Prop::INTEGER, "ID of the message to remove, taken from message_id attribute in <message> tag."),
        )],
        required: vec!["message_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                if !opened.is_admin {
                    return Ok("Error: you are not an administrator of this chat.".to_string());
                }
                let message_id = ctx.arg_i64("message_id").ok_or("message_id is required")? as i32;
                app.tg.delete_messages(opened.chat_id, &[message_id]).await?;
                Ok(format!("Message {message_id} was removed successfully."))
            })
        }),
    }
}

/// tools::groupAdminBanUser.
pub fn group_admin_ban_user(app: Arc<App>, opened: Arc<OpenedChat>) -> Tool {
    Tool {
        name: "group_admin_ban_user".into(),
        description: format!(
            "Bans a user from the \"{}\" chat as it's administrator (makes the vermins forbidden to \
communicate in this chat). You should use this when someone is trying to do something \
prohibited by the chat rules.",
            opened.title
        ),
        properties: vec![(
            "user_id".into(),
            Prop::new(Prop::INTEGER, "ID of the user to ban, taken from the sender_id attribute in <message> tag."),
        )],
        required: vec!["user_id".into()],
        handler: Arc::new(move |ctx| {
            let app = app.clone();
            let opened = opened.clone();
            Box::pin(async move {
                if !opened.is_admin {
                    return Ok("Error: you are not an administrator of this chat.".to_string());
                }
                let user_id = ctx.arg_i64("user_id").ok_or("user_id is required")?;
                let chat = app.tg.peer_ref(opened.chat_id).await?;
                let user = app.tg.peer_ref(user_id).await?;
                let result = "User {user_id} was banned successfully.";
                app.tg
                    .client
                    .kick_participant(chat, user)
                    .await
                    .map_err(err_display)?;
                Ok(result.replace("{user_id}", &user_id.to_string()))
            })
        }),
    }
}

/// Assembles the desktop (non-chat) tool set for one worker iteration
/// (kuni's `updateTools`; async because the tool availability depends on config).
pub async fn build_desktop_tool_list(app: &Arc<App>, ctx: &Ctx) -> Vec<Tool> {
    let cfg = app.config.get().await;
    let mut tools = vec![
        ask(app.clone()),
        get_telegram_chats(app.clone()),
        search_chats(app.clone()),
        search_messages(app.clone()),
        view_messages_around(app.clone()),
        remove_and_ban_chat(app.clone()),
        open_chat_by_id(app.clone()),
        get_chat_admins(app.clone()),
    ];
    if cfg.misc.can_join_chats {
        tools.push(join_chat_by_link(app.clone()));
    }
    let _ = ctx;
    tools
}

/// Builds the notification's action tool set: the single `open` tool for the chat the
/// event came from (kuni's `updateNewMessage` handler).
pub fn build_open_notification_actions(app: &Arc<App>, chat_id: i64, title: &str) -> OpenAITools {
    let tools = OpenAITools::new();
    let app2 = app.clone();
    tools.insert(vec![Tool {
        name: "open".into(),
        description: format!(
            "Open \"{title}\" chat. Use this if you'd like to reply or see messages."
        ),
        properties: vec![],
        required: vec![],
        handler: Arc::new(move |ctx| {
            let app = app2.clone();
            let chat_id = chat_id;
            Box::pin(async move { crate::app::open_telegram_chat(&app, &ctx, chat_id).await })
        }),
    }]);
    tools
}

/// Assembles the chat tool set for an opened chat (kuni's `naxyi:` block in
/// `llmuiOpenTelegramChat` + the per-kind early returns).
///
/// `is_member` mirrors kuni's getChatMember check (false for a channel/group the
/// account has not joined). Call order of the tool set matches kuni's OpenAITools{}
/// initializer (send_telegram_message first).
pub fn build_chat_tools(
    app: Arc<App>,
    opened: Arc<OpenedChat>,
    cfg: &crate::config::Config,
    is_member: bool,
) -> Vec<Tool> {
    // Mirror the official Telegram client behavior: if you open a group chat/channel
    // you haven't joined, you only see a "Join" button (plus the ability to react to
    // messages) instead of a text field.
    if !matches!(opened.kind, ChatKind::Private) && !is_member {
        let mut tools = vec![react_with_emoji(app.clone(), opened.clone())];
        if cfg.misc.can_join_chats {
            tools.push(join_chat(app.clone(), opened.clone()));
        }
        return tools;
    }

    // Member channel: only react + forward (kuni's chatTypeSupergroup.is_channel_
    // branch — `#send_telegram_message tool is not available`).
    if opened.kind == ChatKind::Channel {
        return vec![
            react_with_emoji(app.clone(), opened.clone()),
            forward_message(app.clone(), opened.clone()),
        ];
    }

    let mut tools = vec![
        send_telegram_message(app.clone(), opened.clone()),
        get_chat_photo(app.clone(), opened.clone()),
        react_with_emoji(app.clone(), opened.clone()),
        remove_message(app.clone(), opened.clone()),
        edit_message_text(app.clone(), opened.clone()),
        forward_message(app.clone(), opened.clone()),
    ];

    if matches!(opened.kind, ChatKind::Group) {
        if cfg.misc.can_leave_chats {
            tools.push(leave_chat(app.clone(), opened.clone()));
        }
        if opened.is_admin {
            tools.push(group_admin_remove_message(
                app.clone(),
                opened.clone(),
            ));
            tools.push(group_admin_ban_user(app.clone(), opened.clone()));
            // tools::groupAdminSetUserTag lands with the diary/tags phase.
        }
    }

    tools
}
