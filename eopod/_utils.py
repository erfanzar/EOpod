# Copyright 2025 The EasyDeL/eopod Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import configparser
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.theme import Theme

console = Console(theme=Theme({"info": "cyan", "warning": "yellow", "error": "red", "success": "green"}))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def find_eopod_in_current_env() -> pathlib.Path:
    """Return the absolute path to 'eopod' inside the current venv (or system)."""
    if eopod := os.getenv("EOPOD_EXECUTABLE_PATH"):
        return pathlib.Path(eopod).expanduser().resolve()

    bin_dir = pathlib.Path(sys.executable).parent
    eopod_path = bin_dir / "eopod"

    if eopod_path.is_file():
        return eopod_path

    eopod_path = shutil.which("eopod")
    if eopod_path:
        return pathlib.Path(eopod_path)

    raise FileNotFoundError("eopod executable could not be located")


EOPOD_PATH = str(find_eopod_in_current_env())
PYTHON_PATH = str(sys.executable)


def list2cmdline(seq):
    """Convert a sequence to a command line string (Windows compatible)."""
    result = []
    for arg in map(os.fsdecode, seq):
        bs_buf = []
        if result:
            result.append(" ")
        needquote = (" " in arg) or ("\t" in arg) or not arg
        if needquote:
            result.append('"')
        for c in arg:
            if c == "\\":
                bs_buf.append(c)
            elif c == '"':
                result.append("\\" * len(bs_buf) * 2)
                bs_buf = []
                result.append('\\"')
            else:
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)
        if bs_buf:
            result.extend(bs_buf)
        if needquote:
            result.extend(bs_buf)
            result.append('"')
    return "".join(result)


class TPUManager:
    def __init__(self, project_id: str, zone: str, tpu_name: str):
        self.project_id = project_id
        self.zone = zone
        self.tpu_name = tpu_name

    async def get_status(self) -> dict:
        """Get TPU status information."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "describe",
            self.tpu_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--format=json",
        ]

        console.print("[yellow]Fetching TPU status...[/yellow]")
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            status = json.loads(stdout)
            console.print(f"TPU state: [success]{status.get('state', 'UNKNOWN')}[/]")
            return status
        else:
            error_message = stderr.decode()
            console.print(f"[red]Failed to get TPU status[/]: {error_message}")
            raise RuntimeError(f"Failed to get TPU status: {error_message}")

    async def execute_command(
        self, command: str, worker: str = "all", stream: bool = False, background: bool = False
    ) -> tuple:
        """Execute a command on TPU VM workers."""
        if background:
            command = f"nohup {command} > /tmp/nohup.out 2>&1 & echo $!"

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.tpu_name,
            f"--zone={self.zone}",
            f"--worker={worker}",
            f"--project={self.project_id}",
            f"--command={command}",
        ]

        console.print(f"Executing command on worker {worker}: [info]{command}[/]")

        if stream:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                exit_code = os.system(list2cmdline(cmd))
                if exit_code == 0:
                    progress.print("[blue]Command completed successfully[/]")
                else:
                    progress.print("[red]Command failed[/]")
                return exit_code, "", ""
        else:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                if background:
                    pid = stdout.decode().strip()
                    console.print(f"Background process started with PID: [success]{pid}[/]")
                    return process.returncode, pid, stderr.decode()
                else:
                    console.print("[success]Command completed successfully[/]")
                    return process.returncode, stdout.decode(), stderr.decode()
            else:
                console.print(f"[red]Command failed: {stderr.decode()}[/]")
                return process.returncode, stdout.decode(), stderr.decode()

    async def get_tpu_details(self) -> dict:
        """Fetch detailed information about the TPU."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            self.tpu_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--format=json",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return json.loads(stdout)
        else:
            error_message = stderr.decode()
            console.print(f"[red]Failed to fetch TPU details[/]: {error_message}")
            raise RuntimeError(f"Failed to fetch TPU details: {error_message}")

    async def get_internal_ips(self) -> dict:
        """Get internal IP addresses of TPU workers."""
        try:
            tpu_details = await self.get_tpu_details()
            network_endpoints = tpu_details.get("networkEndpoints", [])
            if not network_endpoints:
                console.print("[yellow]No network endpoints found for the TPU[/yellow]")
                return {}

            internal_ips = {}
            for idx, endpoint in enumerate(network_endpoints):
                worker_id = f"worker-{idx}"
                internal_ip = endpoint.get("ipAddress")
                if internal_ip:
                    internal_ips[worker_id] = internal_ip
                else:
                    console.print(f"[yellow]No internal IP found for {worker_id}[/yellow]")

            return internal_ips
        except Exception as e:
            console.print(f"[red]Error fetching internal IPs: {e!s}[/red]")
            raise

    async def get_external_ips(self) -> str:
        """Get external IP addresses of TPU workers."""
        try:
            cmd = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                self.tpu_name,
                f"--zone={self.zone}",
                f"--project={self.project_id}",
                '--format="value(networkEndpoints[].accessConfig.externalIp)"',
            ]
            string_command = " ".join(cmd)
            process = subprocess.run(string_command, shell=True, capture_output=True, text=True)
            return process.stdout.replace(";", ",").strip()
        except Exception as e:
            console.print(f"[red]Error fetching external IPs: {e!s}[/red]")
            raise

    def format_ips_comma_separated(self, ips: dict) -> str:
        """Format IP addresses as a comma-separated string."""
        return ",".join(ips.values())

    def display_ips(self, ips: dict, ip_type: str, output_format: str = "table"):
        """Display IP addresses in the specified format."""
        if not ips:
            console.print(f"[yellow]No {ip_type} IPs found[/yellow]")
            return

        if output_format == "comma":
            comma_separated_ips = self.format_ips_comma_separated(ips)
            console.print(f"{comma_separated_ips}")
        else:
            table = Table(title=f"{ip_type.capitalize()} IP Addresses")
            table.add_column("Worker", style="cyan")
            table.add_column(f"{ip_type.capitalize()} IP", style="info")
            for worker, ip in ips.items():
                table.add_row(worker, ip)
            console.print(table)


def async_command(fn):
    """Decorator to run async functions in CLI commands."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


async def run_command(command, capture_output=False):
    """Run a command locally and return the result."""
    process = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
    )

    if capture_output:
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode()
            raise Exception(f"Command failed with exit code {process.returncode}: {error_msg}")
        return stdout.decode()
    else:
        await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Command failed with exit code {process.returncode}")
        return None


class EOConfig:
    """Configuration manager for eopod."""

    def __init__(self):
        self.config_dir = Path.home() / ".eopod"
        self.config_file = self.config_dir / "config.ini"
        self.history_file = self.config_dir / "history.yaml"
        self.error_log_file = self.config_dir / "error_log.yaml"
        self.log_file = self.config_dir / "eopod.log"
        self.ensure_config_dir()
        self.config = self.load_config()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                RotatingFileHandler(self.log_file, maxBytes=1024 * 1024, backupCount=5),
            ],
        )

    def ensure_config_dir(self):
        """Create configuration directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self):
        """Load configuration from file."""
        config = configparser.ConfigParser()
        if self.config_file.exists():
            config.read(self.config_file)
        return config

    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            self.config.write(f)

    def get_credentials(self):
        """Get stored GCP credentials."""
        if "DEFAULT" not in self.config:
            return None, None, None
        return (
            self.config["DEFAULT"].get("project_id"),
            self.config["DEFAULT"].get("zone"),
            self.config["DEFAULT"].get("tpu_name"),
        )

    def save_command_history(self, command: str, status: str, output: str):
        """Save command to history."""
        history = []
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                history = yaml.safe_load(f) or []

        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "status": status,
                "output": output[:500],  # Truncate long outputs
            }
        )

        history = history[-100:]  # Keep only last 100 entries

        with open(self.history_file, "w") as f:
            yaml.dump(history, f)

    def save_error_log(self, command: str, error: str):
        """Save error details to error log."""
        error_log = []
        if self.error_log_file.exists():
            with open(self.error_log_file, "r") as f:
                try:
                    error_log = yaml.safe_load(f) or []
                except yaml.YAMLError as e:
                    console.print(f"[red]Error loading error log: {e}[/red]")
                    error_log = []

        error_log.append({"timestamp": datetime.now().isoformat(), "command": command, "error": error})

        error_log = error_log[-50:]  # Keep only last 50 errors

        with open(self.error_log_file, "w") as f:
            yaml.dump(error_log, f)
