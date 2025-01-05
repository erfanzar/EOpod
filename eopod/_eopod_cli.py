# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
# limitations under the License.#

import asyncio
import configparser
from functools import wraps
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class AsyncContext:
	def __init__(self, delay):
		self.delay = delay

	async def __aenter__(self):
		await asyncio.sleep(self.delay)
		return self.delay

	async def __aexit__(self, exc_type, exc, tb):
		await asyncio.sleep(self.delay)


TestAsyncContext = AsyncContext


def async_command(fn):
	@wraps(fn)
	def wrapper(*args, **kwargs):
		return asyncio.run(fn(*args, **kwargs))

	return wrapper


class EOConfig:
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
		logging.basicConfig(
			level=logging.INFO,
			format="%(message)s",
			handlers=[
				RichHandler(rich_tracebacks=True),
				RotatingFileHandler(
					self.log_file,
					maxBytes=1024 * 1024,
					backupCount=5,
				),
			],
		)

	def ensure_config_dir(self):
		self.config_dir.mkdir(parents=True, exist_ok=True)

	def load_config(self):
		config = configparser.ConfigParser()
		if self.config_file.exists():
			config.read(self.config_file)
		return config

	def save_config(self):
		with open(self.config_file, "w") as f:
			self.config.write(f)

	def get_credentials(self):
		if "DEFAULT" not in self.config:
			return None, None, None
		return (
			self.config["DEFAULT"].get("project_id"),
			self.config["DEFAULT"].get("zone"),
			self.config["DEFAULT"].get("tpu_name"),
		)

	def save_command_history(self, command: str, status: str, output: str):
		history = []
		if self.history_file.exists():
			with open(self.history_file, "r") as f:
				history = yaml.safe_load(f) or []

		history.append(
			{
				"timestamp": datetime.now().isoformat(),
				"command": command,
				"status": status,
				"output": output[:500],
			}
		)

		# Keep only last 100 commands in history
		history = history[-100:]

		with open(self.history_file, "w") as f:
			yaml.dump(history, f)

	def save_error_log(self, command: str, error: str):
		"""Saves error details to a separate error log."""
		error_log = []
		if self.error_log_file.exists():
			with open(self.error_log_file, "r") as f:
				try:
					error_log = yaml.safe_load(f) or []
				except yaml.YAMLError as e:
					console.print(f"[red]Error loading error log: {e}[/red]")
					error_log = []

		error_log.append(
			{
				"timestamp": datetime.now().isoformat(),  # Add timestamp here
				"command": command,
				"error": error,
			}
		)

		# Keep only last 50 errors
		error_log = error_log[-50:]

		with open(self.error_log_file, "w") as f:
			yaml.dump(error_log, f)


class TPUManager:
	def __init__(self, project_id: str, zone: str, tpu_name: str):
		self.project_id = project_id
		self.zone = zone
		self.tpu_name = tpu_name

	async def get_status(self) -> dict:
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

		process = await asyncio.create_subprocess_exec(
			*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
		)

		stdout, stderr = await process.communicate()
		if process.returncode == 0:
			return json.loads(stdout)
		else:
			error_message = stderr.decode()
			logging.error(f"Failed to get TPU status: {error_message}")
			raise RuntimeError(f"Failed to get TPU status: {error_message}")

	async def execute_command(self, command: str, worker: str = "all") -> tuple:
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

		process = await asyncio.create_subprocess_exec(
			*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
		)

		stdout, stderr = await process.communicate()
		return process.returncode, stdout.decode(), stderr.decode()


@click.group()
def cli():
	"""EOpod - Enhanced TPU Command Runner"""
	pass


@cli.command()
@click.option("--project-id", required=True, help="Google Cloud Project ID")
@click.option("--zone", required=True, help="Google Cloud Zone")
@click.option("--tpu-name", required=True, help="TPU Name")
def configure(project_id, zone, tpu_name):
	"""Configure EOpod with your Google Cloud details"""
	config = EOConfig()
	if "DEFAULT" not in config.config:
		config.config["DEFAULT"] = {}

	config.config["DEFAULT"]["project_id"] = project_id
	config.config["DEFAULT"]["zone"] = zone
	config.config["DEFAULT"]["tpu_name"] = tpu_name
	config.save_config()
	console.print("[green]Configuration saved successfully![/green]")


@cli.command()
@click.argument("command")
@click.option(
	"--worker",
	default="all",
	help='Specific worker or "all"',
)
@click.option(
	"--retry",
	default=3,
	help="Number of retries for failed commands",
)
@click.option(
	"--delay",
	default=5,
	help="Delay between retries in seconds",
)
@click.option(
	"--timeout",
	default=300,
	help="Command timeout in seconds",
)
@click.option(
	"--interactive",
	is_flag=True,
	help="Run command in interactive mode (experimental)",
)
@click.option(
	"--stream",
	is_flag=True,
	help="Stream the output from the specified worker(s)",
)
@click.option(
	"--nohup",
	is_flag=True,
	help="Run the command in the background, detached from the session",
)
@async_command
async def run(command, worker, retry, delay, timeout, interactive, stream, nohup):
	"""Run a command on TPU VM with advanced features"""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)

	if interactive:
		console.print(
			"[yellow]Interactive mode is experimental. Use with caution.[/yellow]"
		)
		cmd = [
			"gcloud",
			"compute",
			"tpus",
			"tpu-vm",
			"ssh",
			tpu_name,
			f"--zone={zone}",
			f"--worker={worker}",
			f"--project={project_id}",
		]
		# Start an interactive process (no command specified)
		process = await asyncio.create_subprocess_exec(*cmd)
		await process.wait()  # Wait for the process to complete
		return

	if stream:
		await tpu.stream_command(command, worker)
		return

	if nohup:
		# Wrap the command with nohup and redirect output/errors to files
		nohup_command = f"nohup {command} > {tpu_name}_{worker}_output.log 2> {tpu_name}_{worker}_error.log &"
		returncode, _, _ = await tpu.execute_command(
			nohup_command, worker, capture_output=False
		)
		if returncode == 0:
			console.print(
				f"[green]Command '{command}' started in the background on worker(s) {worker}. "
				f"Output and errors redirected to {tpu_name}_{worker}_output.log and {tpu_name}_{worker}_error.log[/green]"
			)
		else:
			console.print(
				f"[red]Failed to start command in the background on worker(s) {worker}.[/red]"
			)
		return

	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
	) as progress:
		task = progress.add_task(
			description=f"Executing command: {command} (Attempt 1)", total=None
		)

		for attempt in range(1, retry + 1):
			try:
				returncode, stdout, stderr = await asyncio.wait_for(
					tpu.execute_command(command, worker), timeout=timeout
				)

				if returncode == 0:
					progress.update(
						task,
						description="[green]Command completed successfully![/green]",
					)
					console.print("\nOutput:")
					console.print(stdout)
					config.save_command_history(command, "success", stdout)
					break
				else:
					progress.update(
						task,
						description=f"[red]Attempt {attempt} failed:[/red] {stderr[:100]}...",
					)
					console.print(f"[red]Attempt {attempt} failed:[/red] {stderr}")
					config.save_error_log(command, stderr)

			except asyncio.TimeoutError:
				progress.update(
					task,
					description=f"[red]Command timed out after {timeout} seconds (attempt {attempt})[/red]",
				)
				console.print(
					f"[red]Command timed out after {timeout} seconds (attempt {attempt})[/red]"
				)
				config.save_error_log(command, "Command timed out")

			except Exception as e:
				progress.update(
					task, description=f"[red]Error (attempt {attempt}):[/red] {str(e)}"
				)
				console.print(f"[red]Error (attempt {attempt}):[/red] {str(e)}")
				config.save_error_log(command, str(e))
				break

			if attempt < retry:
				progress.update(
					task,
					description=f"Executing command: {command} (Attempt {attempt + 1})",
				)
				time.sleep(delay)
			else:
				progress.update(
					task,
					description=f"[red]Command failed after {retry} attempts[/red]",
				)


@cli.command()
@async_command
async def status():
	"""Show TPU status and information"""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	try:
		tpu = TPUManager(project_id, zone, tpu_name)
		status = await tpu.get_status()

		table = Table(title="TPU Status")
		table.add_column("Property")
		table.add_column("Value")

		table.add_row("Name", status.get("name", ""))
		table.add_row("State", status.get("state", ""))
		table.add_row("Type", status.get("acceleratorType", ""))
		table.add_row("Network", status.get("network", ""))
		table.add_row("API Version", status.get("apiVersion", ""))

		console.print(table)

	except RuntimeError as e:
		console.print(f"[red]{e}[/red]")


@cli.command()
def history():
	"""Show command execution history"""
	config = EOConfig()

	if not config.history_file.exists():
		console.print("No command history found.")
		return

	with open(config.history_file, "r") as f:
		history = yaml.safe_load(f) or []

	table = Table(title="Command History")
	table.add_column("Timestamp")
	table.add_column("Command")
	table.add_column("Status")
	table.add_column("Output (truncated)")

	for entry in history[-15:]:
		table.add_row(
			entry["timestamp"], entry["command"], entry["status"], entry["output"]
		)

	console.print(table)


@cli.command()
def errors():
	"""Show recent command execution errors."""
	config = EOConfig()

	if not config.error_log_file.exists():
		console.print("No error log found.")
		return

	with open(config.error_log_file, "r") as f:
		try:
			error_log = yaml.safe_load(f) or []
		except yaml.YAMLError as e:
			console.print(f"[red]Error loading error log: {e}[/red]")
			return

	table = Table(title="Error Log", style="red")
	table.add_column("Timestamp")
	table.add_column("Command")
	table.add_column("Error")

	for entry in error_log:
		table.add_row(entry["timestamp"], entry["command"], entry["error"][:200])

	console.print(table)


@cli.command()
def show_config():
	"""Show current configuration"""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if all([project_id, zone, tpu_name]):
		table = Table(title="Current Configuration")
		table.add_column("Setting")
		table.add_column("Value")

		table.add_row("Project ID", project_id)
		table.add_row("Zone", zone)
		table.add_row("TPU Name", tpu_name)

		console.print(table)
	else:
		console.print(
			"[red]No configuration found. Please run 'eopod configure' first.[/red]"
		)


def main():
	"""
	Main entry point for the EOpod CLI.
	"""
	try:
		asyncio.run(cli())
	except click.exceptions.Exit as e:
		if e.exit_code != 0:
			console.print(f"[red]Error:[/red] Command failed with exit code {e.exit_code}")
			logging.exception("Click command failed")
	except Exception as e:
		console.print(f"[red]Unexpected Error:[/red] {str(e)}")
		logging.exception("An unexpected error occurred")


if __name__ == "__main__":
	main()
