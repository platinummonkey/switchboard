//! `switchboard-local` — developer-side proxy for Switchboard.
//!
//! Subcommands:
//! - `init`   — interactive setup wizard
//! - `start`  — start the proxy in the foreground
//! - `status` — print connectivity and configuration status
//! - `stop`   — (not implemented) show instructions to kill the process

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};

use switchboard_local::auth;
use switchboard_local::config::{
    self, AuthConfig, IdentityConfig, JwtAuthConfig, LocalConfig, LocalListenConfig, ModelConfig,
    ServerConfig, default_config_path,
};
use switchboard_local::error::LocalError;
use switchboard_local::server::LocalServer;

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    name = "switchboard-local",
    about = "switchboard-local: developer-side proxy for Switchboard",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Interactive setup: writes ~/.switchboard/config.toml
    Init,
    /// Start the local proxy in the foreground (logs to stdout)
    Start,
    /// Show current status and connectivity
    Status,
    /// Stop the running proxy (not implemented — kill the process)
    Stop,
}

// ── Entry point ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => cmd_init().await?,
        Commands::Start => cmd_start().await?,
        Commands::Status => cmd_status().await?,
        Commands::Stop => cmd_stop(),
    }

    Ok(())
}

// ── `init` ────────────────────────────────────────────────────────────────────

async fn cmd_init() -> Result<(), LocalError> {
    use std::io::{BufRead, Write};

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    macro_rules! prompt {
        ($default:expr, $($fmt:tt)*) => {{
            print!($($fmt)*);
            if let Some(default) = $default {
                print!(" [{}]", default);
            }
            print!(": ");
            stdout.flush().map_err(LocalError::Io)?;
            let mut line = String::new();
            stdin.lock().read_line(&mut line).map_err(LocalError::Io)?;
            let trimmed = line.trim().to_owned();
            if trimmed.is_empty() {
                $default.map(|s: &str| s.to_owned()).unwrap_or_default()
            } else {
                trimmed
            }
        }};
    }

    println!("=== switchboard-local setup ===");
    println!();

    // Server URL
    let server_url = prompt!(
        Some("https://switchboard.internal:8080"),
        "Switchboard server URL"
    );

    // Auth method
    println!();
    println!("Auth method options: api_key, jwt, skip");
    let auth_method_raw = prompt!(Some("api_key"), "Auth method");
    let auth_method = match auth_method_raw.as_str() {
        "api_key" | "jwt" | "skip" => auth_method_raw.clone(),
        other => {
            eprintln!("Unknown auth method '{other}', defaulting to 'api_key'");
            "api_key".to_owned()
        }
    };

    let mut api_key: Option<String> = None;
    let mut jwt_token: Option<String> = None;
    let mut jwt_token_command: Option<String> = None;

    match auth_method.as_str() {
        "api_key" => {
            let key = prompt!(None::<&str>, "API key");
            if !key.is_empty() {
                api_key = Some(key);
            }
        }
        "jwt" => {
            println!("Provide a static JWT token, or a shell command to fetch one.");
            let token_or_cmd = prompt!(None::<&str>, "Token (leave blank to enter a command)");
            if token_or_cmd.is_empty() {
                let cmd = prompt!(None::<&str>, "Token command (shell)");
                if !cmd.is_empty() {
                    jwt_token_command = Some(cmd);
                }
            } else {
                jwt_token = Some(token_or_cmd);
            }
        }
        _ => {}
    }

    // Identity
    println!();
    let os_user = std::env::var("USER").unwrap_or_else(|_| "unknown".into());
    let user_input = prompt!(
        Some(os_user.as_str()),
        "Your user identity (e.g. you@example.com)"
    );
    let identity_user = if user_input.is_empty() {
        None
    } else {
        Some(user_input)
    };
    let team_input = prompt!(Some(""), "Your team (optional, press Enter to skip)");
    let identity_team = if team_input.is_empty() {
        None
    } else {
        Some(team_input)
    };

    // Build config
    let cfg = LocalConfig {
        server: ServerConfig { url: server_url },
        auth: AuthConfig {
            method: if auth_method == "skip" {
                "api_key".to_owned()
            } else {
                auth_method
            },
            api_key,
            jwt: JwtAuthConfig {
                token: jwt_token,
                token_command: jwt_token_command,
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        },
        identity: IdentityConfig {
            user: identity_user,
            team: identity_team,
        },
        local: LocalListenConfig::default(),
        model: ModelConfig::default(),
    };

    // Write config
    let config_path = default_config_path();
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).map_err(LocalError::Io)?;
    }
    let toml_str = toml::to_string_pretty(&cfg)
        .map_err(|e| LocalError::Config(format!("failed to serialize config: {e}")))?;
    std::fs::write(&config_path, &toml_str).map_err(LocalError::Io)?;

    println!();
    println!("Config written to {}", config_path.display());
    println!("Run `switchboard-local start` to start the proxy.");

    Ok(())
}

// ── `start` ───────────────────────────────────────────────────────────────────

async fn cmd_start() -> Result<(), LocalError> {
    tracing_subscriber::fmt::init();

    let config = config::load_or_default()?;
    tracing::info!(
        listen = %config.local.listen,
        server = %config.server.url,
        auth = %config.auth.method,
        "switchboard-local starting"
    );

    let server = LocalServer::new(Arc::new(config)).await?;
    server.run().await
}

// ── `status` ──────────────────────────────────────────────────────────────────

async fn cmd_status() -> Result<(), LocalError> {
    let config = config::load_or_default()?;

    println!("=== switchboard-local status ===");
    println!("Listen addr : {}", config.local.listen);
    println!("Server URL  : {}", config.server.url);
    println!("Auth method : {}", config.auth.method);
    println!("Default model: {}", config.model.default);

    // Attempt connectivity check.
    let auth_manager = auth::LocalAuthManager::new(&config.auth).await;

    match auth_manager {
        Err(e) => {
            println!("Auth        : ERROR — {e}");
        }
        Ok(manager) => {
            match manager.get_header().await {
                Err(e) => {
                    println!("Auth        : ERROR — {e}");
                }
                Ok((name, value)) => {
                    // Try a GET /health against the server.
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(5))
                        .build();

                    match client {
                        Err(e) => println!("HTTP client : ERROR — {e}"),
                        Ok(client) => {
                            let health_url =
                                format!("{}/health", config.server.url.trim_end_matches('/'));
                            let resp = client.get(&health_url).header(name, value).send().await;

                            match resp {
                                Ok(r) if r.status().is_success() => {
                                    println!("Server      : OK ({})", r.status());
                                }
                                Ok(r) => {
                                    println!("Server      : reachable but returned {}", r.status());
                                }
                                Err(e) => {
                                    println!("Server      : UNREACHABLE — {e}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ── `stop` ────────────────────────────────────────────────────────────────────

fn cmd_stop() {
    println!(
        "switchboard-local does not support a stop command.\n\
         To stop the proxy, find the process and kill it:\n\
         \n    pkill switchboard-local\n\
         \nor find the PID with `pgrep switchboard-local` and use `kill <pid>`."
    );
}
