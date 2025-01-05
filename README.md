# EOpod: Enhanced TPU Command Runner

EOpod is a streamlined command-line tool designed to simplify and enhance the execution of commands on Google Cloud TPU VMs. It provides a user-friendly interface for running commands, managing configurations, and viewing command history, making it easier to work with TPUs in your projects.

## Features

* **Configuration Management:** Easily configure EOpod with your Google Cloud project ID, zone, and TPU name.
* **Command Execution:** Run commands on TPU VMs with advanced features like retries, delays, timeouts, and worker selection.
* **Interactive Mode (Experimental):** Run commands in an interactive SSH session (use with caution).
* **Command History:** View a history of executed commands, their status, and truncated output.
* **Error Logging:** Detailed error logs are maintained for debugging failed commands.
* **Rich Output:** Utilizes the `rich` library for visually appealing and informative output in the console.

## Installation

### Using `pip`

```bash
pip install eopod
```

### Using `poetry`

```bash
poetry add eopod
```

### From Source (using Poetry)

1. Clone the repository:

    ```bash
    git clone https://github.com/erfanzar/EOpod.git
    cd EOpod
    ```

2. Install using Poetry:

    ```bash
    poetry install
    ```

## Usage

### Configuration

Before using EOpod, you need to configure it with your Google Cloud project details:

```bash
eopod configure --project-id <your_project_id> --zone <your_zone> --tpu-name <your_tpu_name>
```

Replace `<your_project_id>`, `<your_zone>`, and `<your_tpu_name>` with your actual project ID, zone, and TPU name.

### Running Commands

You can run commands on your TPU VM using the `run` command:

```bash
eopod run <command> [options]
```

* **`<command>`:** The command to execute on the TPU VM (e.g., `ls`, `python my_script.py`, etc.).
* **`--worker`:**  Specify the worker to run the command on (default: `all`).
* **`--retry`:** Set the number of retries for failed commands (default: 3).
* **`--delay`:** Set the delay in seconds between retries (default: 5).
* **`--timeout`:** Set the command timeout in seconds (default: 300).
* **`--interactive`:** Run the command in an interactive SSH session (experimental).

**Examples:**

* Run `ls -l` on all workers:

    ```bash
    eopod run "ls -l"
    ```

* Run `python train.py` with 5 retries and a 10-second delay:

    ```bash
    eopod run "python train.py" --retry 5 --delay 10
    ```

* Run `nvidia-smi` on worker 0 with a timeout of 60 seconds:

    ```bash
    eopod run "pip install easydel" --worker 0 --timeout 60
    ```

### Viewing Command History

To see a history of executed commands:

```bash
eopod history
```

This will display a table with the timestamp, command, status, and truncated output of the last 15 commands.

### Viewing Error Logs

To view detailed error logs:

```bash
eopod errors
```

This will display a table showing the timestamp, command, and error message for recent failed commands.

### Showing Configuration

To see your current EOpod configuration:

```bash
eopod show-config
```
