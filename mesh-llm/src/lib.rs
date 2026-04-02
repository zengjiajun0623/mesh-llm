mod affinity;
mod api;
pub(crate) mod app;
mod autoupdate;
mod benchmark;
mod cli;
mod election;
mod hardware;
mod launch;
mod mesh;
mod models;
mod moe;
mod nostr;
mod pipeline;
mod plugin;
mod plugin_mcp;
mod plugins;
mod protocol;
mod proxy;
mod rewrite;
mod router;
mod tunnel;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use autoupdate::{latest_release_version, version_newer};
pub use plugins::blackboard;
pub use plugins::blackboard::mcp as blackboard_mcp;

use anyhow::Result;

pub const VERSION: &str = "0.54.0";

pub async fn run() -> Result<()> {
    app::run().await
}
