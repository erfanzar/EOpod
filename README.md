# EOpod: Enhanced TPU Command Runner

EOpod is a command-line tool designed to simplify and enhance interaction with Google Cloud TPU VMs. It provides real-time output streaming, background process management, and robust error handling.

## Features

* **Configuration Management:** Easily configure EOpod with your Google Cloud project ID, zone, and TPU name.
* **Command Execution:** Run commands on TPU VMs with advanced features like retries, delays, timeouts, and worker selection.
* **Interactive Mode (Experimental):** Run commands in an interactive SSH session (use with caution).
* **Command History:** View a history of executed commands, their status, and truncated output.
* **Error Logging:** Detailed error logs are maintained for debugging failed commands.
* **Rich Output:** Utilizes the `rich` library for visually appealing and informative output in the console.

## Installation

```bash
pip install eopod
```

## Configuration

Before using EOpod, configure it with your Google Cloud credentials:

```bash
eopod configure --project-id YOUR_PROJECT_ID --zone YOUR_ZONE --tpu-name YOUR_TPU_NAME
```

## Usage Examples

### Basic Command Execution

Commands are executed with real-time output streaming by default:

```bash
# Simple command
eopod run echo "Hello TPU"

# Run Python script
eopod run python train.py --batch-size 32

# Complex commands with pipes and redirections
eopod run "cat data.txt | grep error > errors.log"

# Commands with multiple arguments
eopod run ls -la /path/to/dir
```

### Background Processes

Run long-running tasks in the background:

```bash
# Start training in background
eopod run python long_training.py --epochs 1000 --background

# Check background processes
eopod check-background

# Check specific process
eopod check-background 12345

# Kill a background process
eopod kill 12345

# Force kill if necessary
eopod kill 12345 --force
```

### Worker-Specific Commands

Execute commands on specific workers:

```bash
# Run on specific worker
eopod run nvidia-smi --worker 0

# Run on all workers (default)
eopod run hostname --worker all
```

### Advanced Options

```bash
# Disable output streaming
eopod run python script.py --no-stream

# Set custom retry count
eopod run python train.py --retry 5

# Set custom retry delay
eopod run python train.py --delay 10

# Set custom timeout
eopod run python train.py --timeout 600
```

### Kill and free TPU process

```bash
# Kill all TPU processes
eopod kill-tpu

# Force kill all TPU processes
eopod kill-tpu --force

# Kill specific PID(s)
eopod kill-tpu --pid 1234 --pid 5678

# Kill processes on specific worker
eopod kill-tpu --worker 0
```

### Viewing History and Logs

```bash
# View command history
eopod history

# View error logs
eopod errors

# View current configuration
eopod show-config
```

## Command Reference

### Main Commands

* `run`: Execute commands on TPU VM

  ```bash
  eopod run [OPTIONS] COMMAND [ARGS]...
  ```

  Options:
  * `--worker TEXT`: Specific worker or "all" (default: "all")
  * `--retry INTEGER`: Number of retries for failed commands (default: 3)
  * `--delay INTEGER`: Delay between retries in seconds (default: 5)
  * `--timeout INTEGER`: Command timeout in seconds (default: 300)
  * `--no-stream`: Disable output streaming
  * `--background`: Run command in background

* `configure`: Set up EOpod configuration

  ```bash
  eopod configure --project-id ID --zone ZONE --tpu-name NAME
  ```

* `status`: Check TPU status

  ```bash
  eopod status
  ```

* `check-background`: Check background processes

  ```bash
  eopod check-background [PID]
  ```

* `kill`: Kill background processes

  ```bash
  eopod kill PID [--force]
  ```

### Utility Commands

* `history`: View command execution history
* `errors`: View error logs
* `show-config`: Display current configuration

## File Locations

* Configuration: `~/.eopod/config.ini`
* Command history: `~/.eopod/history.yaml`
* Error logs: `~/.eopod/error_log.yaml`
* Application logs: `~/.eopod/eopod.log`

## Error Handling

EOpod includes built-in error handling and retry mechanisms:

* Automatic retry for failed commands
* Timeout handling
* Detailed error logging
* Rich error output

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
