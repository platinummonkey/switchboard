//! `switchboard-local` — developer-side proxy for Switchboard.
//!
//! Subcommands:
//! - `init`   — interactive setup wizard
//! - `start`  — start the proxy in the foreground
//! - `status` — print connectivity and configuration status
//! - `stop`   — send SIGTERM to the running proxy via PID file

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
    /// Stop the running proxy via PID file (sends SIGTERM)
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
        Commands::Stop => cmd_stop().await?,
    }

    Ok(())
}

// ── PID file helpers ──────────────────────────────────────────────────────────

/// Returns the path to the PID file: `~/.switchboard/switchboard-local.pid`.
pub(crate) fn pid_file_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".switchboard").join("switchboard-local.pid")
}

/// RAII guard that removes the PID file when dropped (best-effort).
pub(crate) struct PidGuard(pub std::path::PathBuf);

impl Drop for PidGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
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

    // Probe the server URL to verify connectivity.
    println!();
    println!("Testing connectivity to {} ...", cfg.server.url);

    let test_url = format!("{}/health", cfg.server.url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| LocalError::Config(format!("failed to build HTTP client: {e}")))?;

    match client.get(&test_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✓ Server reachable ({})", resp.status());
        }
        Ok(resp) => {
            println!("⚠ Server returned {} — check configuration.", resp.status());
        }
        Err(e) => {
            println!("⚠ Could not reach server: {e}");
            println!("  Config saved anyway. You can retry with `switchboard-local status`.");
        }
    }

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

    // Write PID file.
    let pid_path = pid_file_path();
    let pid = std::process::id();
    std::fs::write(&pid_path, pid.to_string()).map_err(|e| {
        LocalError::Config(format!(
            "failed to write PID file {}: {e}",
            pid_path.display()
        ))
    })?;
    tracing::info!(pid = pid, path = %pid_path.display(), "PID file written");

    // Remove PID file on exit (best-effort).
    let pid_path_clone = pid_path.clone();
    let _pid_guard = PidGuard(pid_path_clone);

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

async fn cmd_stop() -> Result<(), LocalError> {
    let pid_path = pid_file_path();

    if !pid_path.exists() {
        println!("No PID file found at {}.", pid_path.display());
        println!("switchboard-local may not be running, or was started without this version.");
        println!("To stop manually: pkill switchboard-local");
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_path)
        .map_err(|e| LocalError::Config(format!("failed to read PID file: {e}")))?;
    let pid: u32 = pid_str
        .trim()
        .parse()
        .map_err(|_| LocalError::Config(format!("invalid PID in file: '{}'", pid_str.trim())))?;

    #[cfg(unix)]
    {
        // Send SIGTERM to the process.
        let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if result == 0 {
            println!("Sent SIGTERM to switchboard-local (PID {pid}).");
            // Give it 2 seconds then check if still running.
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let still_running = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
            if still_running {
                println!("Process still running after 2s. Send SIGKILL? (run: kill -9 {pid})");
            } else {
                println!("Process stopped.");
                let _ = std::fs::remove_file(&pid_path);
            }
        } else {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(LocalError::Config(format!(
                    "permission denied sending signal to PID {pid}"
                )));
            }
            // ESRCH — no such process: stale PID file
            println!("Process {pid} not found (stale PID file). Cleaning up.");
            let _ = std::fs::remove_file(&pid_path);
        }
    }

    #[cfg(not(unix))]
    {
        println!("Stop via PID file is only supported on Unix. To stop: kill {pid}");
    }

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_file_path_under_home() {
        let path = pid_file_path();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.ends_with(".switchboard/switchboard-local.pid"),
            "expected path ending with .switchboard/switchboard-local.pid, got: {path_str}"
        );
    }

    #[test]
    fn test_pid_guard_removes_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.pid");
        std::fs::write(&file_path, "12345").unwrap();
        assert!(file_path.exists(), "file should exist before drop");
        {
            let _guard = PidGuard(file_path.clone());
        }
        assert!(
            !file_path.exists(),
            "file should be removed after PidGuard is dropped"
        );
    }

    #[tokio::test]
    async fn test_cmd_stop_no_pid_file() {
        // Override HOME to a temp dir where no PID file exists.
        let dir = tempfile::tempdir().unwrap();
        // Safety: single-threaded test, no concurrent env reads.
        unsafe { std::env::set_var("HOME", dir.path()) };
        let result = cmd_stop().await;
        assert!(
            result.is_ok(),
            "cmd_stop should return Ok when no PID file exists"
        );
    }
}
