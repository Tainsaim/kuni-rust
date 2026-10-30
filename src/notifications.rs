//! A port of kuni-master's `src/NotificationManager.*` with tokio primitives.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};

use tokio::sync::{mpsc, Mutex};

use crate::tools::OpenAITools;

/// kuni's `NotificationManager::Notification`.
#[derive(Clone)]
pub struct Notification {
    /// Natural-language message passed to the LLM. Kuni appends "\nCurrent time: ..."
    /// right before the worker processes it — done in `RunningWorker::next`.
    pub message: String,
    /// Related tools available to the LLM when processing this notification
    /// (e.g. the `open` tool of the chat the event came from).
    pub actions: OpenAITools,
    /// Higher priority notifications are processed earlier (papik: 1000, pinned: 100).
    pub priority: i64,
    /// Freeform pin token that "poisons" the worker processing it (e.g. `<chat id="123" />`),
    /// routing subsequent notifications of the same chat to that (context-rich) worker.
    pub pin: Option<String>,
}

struct WorkerSlot {
    /// Pins adopted by this worker (kuni's `workerPins` set).
    pins: Arc<StdMutex<HashSet<String>>>,
    /// Wake-up token channel; tokens are *hints*: a worker that finds nothing for it
    /// simply goes back to sleep.
    wake: mpsc::UnboundedSender<()>,
}

pub struct NotificationManager {
    /// Priority deque (kuni's `mNotifications`).
    notifications: Mutex<VecDeque<Notification>>,
    workers: StdMutex<Vec<WorkerSlot>>,
}

impl Default for NotificationManager {
    fn default() -> Self {
        Self {
            notifications: Mutex::new(VecDeque::new()),
            workers: StdMutex::new(Vec::new()),
        }
    }
}

/// A registered worker; `next` yields notifications it is allowed to process.
pub struct RunningWorker {
    manager: Arc<NotificationManager>,
    receiver: mpsc::UnboundedReceiver<()>,
    /// Slot index in `NotificationManager::workers` (pins live there).
    slot: usize,
}

impl NotificationManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// kuni's `run` registration: adds a worker slot and returns its runner.
    /// The runner loops on `next` until the manager is dropped.
    pub async fn run(self: &Arc<Self>) -> RunningWorker {
        let (tx, rx) = mpsc::unbounded_channel();
        let slot = {
            let mut workers = self.workers.lock().unwrap();
            workers.push(WorkerSlot {
                pins: Arc::new(StdMutex::new(HashSet::new())),
                wake: tx,
            });
            workers.len() - 1
        };
        RunningWorker {
            manager: self.clone(),
            receiver: rx,
            slot,
        }
    }

    /// kuni's `passNotificationToAI`: stable-insert before the first notification of
    /// lower priority, then wake every worker (woken ones that find nothing sleep on).
    pub async fn pass_notification(&self, notification: Notification) {
        {
            let mut queue = self.notifications.lock().await;
            let at = queue.partition_point(|h| h.priority >= notification.priority);
            queue.insert(at, notification);
        }
        for slot in self.workers.lock().unwrap().iter() {
            let _ = slot.wake.send(());
        }
    }

    /// kuni's `removeNotifications`: drops queued notifications containing `substring`.
    pub async fn remove_notifications(&self, substring: &str) {
        self.notifications
            .lock()
            .await
            .retain(|n| !n.message.contains(substring));
    }

    fn pins_handle(&self, slot: usize) -> Arc<StdMutex<HashSet<String>>> {
        self.workers
            .lock()
            .unwrap()
            .get(slot)
            .expect("worker slot disappeared")
            .pins
            .clone()
    }

    fn slot_pinned_elsewhere(&self, slot: usize, pin: &str) -> bool {
        self.workers
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .any(|(no, w)| no != slot && w.pins.lock().unwrap().contains(pin))
    }

    /// Clears all pins adopted by a worker (kuni `Worker::diaryDumpMessages`).
    pub fn clear_pins(&self, slot: usize) {
        self.pins_handle(slot).lock().unwrap().clear();
    }

    fn adopt_pin(&self, slot: usize, pin: String) {
        self.pins_handle(slot).lock().unwrap().insert(pin);
    }
}



impl RunningWorker {
    /// Waits until a notification this worker may process is available, returning it
    /// with the "\nCurrent time: {utc}" suffix appended (kuni's `run` loop).
    pub async fn next(&mut self) -> Option<Notification> {
        loop {
            if let Some(mut n) = self.take_next().await {
                let time = jiff::Timestamp::now().strftime("%Y-%m-%d %H:%M:%S UTC").to_string();
                n.message.push_str(&format!("\nCurrent time: {time}"));
                return Some(n);
            }
            // Sleep until a wake token arrives (or the manager is dropped).
            self.receiver.recv().await?;
            // Consume stale tokens; they carry no data.
            while self.receiver.try_recv().is_ok() {}
        }
    }

    /// kuni's `nextNotification` scan: takes the first unpinned notification, or one
    /// pinned to this worker; skips notifications pinned to other workers; adopts the
    /// pin of whatever it takes.
    async fn take_next(&mut self) -> Option<Notification> {
        let mine = self.manager.pins_handle(self.slot);
        let mut queue = self.manager.notifications.lock().await;
        for idx in 0..queue.len() {
            let Some(pin) = queue[idx].pin.clone() else {
                return queue.remove(idx);
            };
            if mine.lock().unwrap().contains(&pin) {
                return queue.remove(idx);
            }
            if self.manager.slot_pinned_elsewhere(self.slot, &pin) {
                // pinned to another worker, skip.
                continue;
            }
            // Adopt the pin: this chat now belongs to this worker.
            self.manager.adopt_pin(self.slot, pin);
            return queue.remove(idx);
        }
        None
    }
}
