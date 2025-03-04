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
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.theme import Theme

console = Console(
	theme=Theme(
		{
			"info": "cyan",
			"warning": "yellow",
			"error": "white",
			"success": "green",
		}
	)
)

logging.basicConfig(
	level=logging.INFO,
	format="%(message)s",
	handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def list2cmdline(seq):
	result = []
	needquote = False
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


def clean_tqdm_output(line: str) -> str:
	"""Clean up TQDM progress bar output to show only the latest state."""
	if "\r" in line:
		# Take only the last progress bar update
		return line.rstrip().split("\r")[-1]
	return line.rstrip()


def is_tqdm_line(line: str) -> bool:
	"""Check if a line contains TQDM progress bar."""
	return "%|" in line and "it/s]" in line


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

		console.print("[yellow]Fetching TPU status...[/yellow]")
		process = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
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
		self,
		command: str,
		worker: str = "all",
		stream: bool = False,
		background: bool = False,
	) -> tuple:
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
				*cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
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
			# gcloud compute tpus describe LLaMA --zone=us-central2-b --project=phd-research-384422 --format=json
			# gcloud compute tpus tpu-vm describe LLaMA --zone=us-central2-b --project=phd-research-384422 --format="value(networkEndpoints[].accessConfig.externalIp)"

	async def get_tpu_details(self) -> dict:
		"""Fetch detailed information about the TPU."""
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
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
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
			console.print(f"[red]Error fetching internal IPs: {str(e)}[/red]")
			raise

	async def get_external_ips(self) -> dict:
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
			process = subprocess.run(
				string_command, shell=True, capture_output=True, text=True
			)
			return process.stdout.replace(";", ",").strip()
		except Exception as e:
			console.print(f"[red]Error fetching external IPs: {str(e)}[/red]")
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
		else:  # Default to table format
			table = Table(title=f"{ip_type.capitalize()} IP Addresses")
			table.add_column("Worker", style="cyan")
			table.add_column(f"{ip_type.capitalize()} IP", style="info")
			for worker, ip in ips.items():
				table.add_row(worker, ip)
			console.print(table)


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
				"timestamp": datetime.now().isoformat(),
				"command": command,
				"error": error,
			}
		)

		error_log = error_log[-50:]

		with open(self.error_log_file, "w") as f:
			yaml.dump(error_log, f)


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
@click.option(
	"--format",
	type=click.Choice(["table", "comma"]),
	default="comma",
	help="Output format: 'table' or 'comma'",
)
@async_command
async def get_internal_ips(format):
	"""Get internal IP addresses of TPU workers."""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()
	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure the tool first using 'eopod configure'[/red]")
		return

	tpu_manager = TPUManager(project_id, zone, tpu_name)
	try:
		internal_ips = await tpu_manager.get_internal_ips()
		tpu_manager.display_ips(internal_ips, "internal", output_format=format)
	except Exception as e:
		console.print(f"[red]Failed to get internal IPs: {str(e)}[/red]")


@cli.command()
@click.option(
	"--format",
	type=click.Choice(["table", "comma"]),
	default="comma",
	help="Output format: 'table' or 'comma'",
)
@async_command
async def get_external_ips(format):
	"""Get external IP addresses of TPU workers."""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()
	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure the tool first using 'eopod configure'[/red]")
		return
	tpu_manager = TPUManager(project_id, zone, tpu_name)
	try:
		external_ips = await tpu_manager.get_external_ips()
		console.print(external_ips)
	except Exception as e:
		console.print(f"[red]Failed to get external IPs: {str(e)}[/red]")


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("cmd_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--retry", default=1, help="Number of retries for failed commands")
@click.option("--delay", default=5, help="Delay between retries in seconds")
@click.option("--timeout", default=-1, help="Command timeout in seconds")
@click.option("--no-stream", is_flag=True, help="Disable output streaming")
@click.option(
	"--background", is_flag=True, help="Run command in background (nohup-like)"
)
@async_command
async def run(cmd_args, worker, retry, delay, timeout, no_stream, background):
	"""Run a command on TPU VM with advanced features"""
	if not cmd_args:
		console.print("[red]No command provided[/red]")
		return

	# Join arguments preserving quotes and spaces
	command = " ".join(cmd_args)
	stream = not no_stream
	if timeout == -1:
		timeout = None
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)

	start_time = datetime.now()
	console.print(f"[cyan]Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")
	console.print(f"[cyan]Executing: {command}[/cyan]")

	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		disable=stream,  # Disable progress bar when streaming
	) as progress:
		task = progress.add_task(
			description=f"Executing command: {command} (Attempt 1)", total=None
		)

		for attempt in range(1, retry + 1):
			try:
				if background:
					# Add more detailed background process handling
					background_cmd = (
						f"nohup {command} > /tmp/nohup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.out "
						"2>&1 & echo $!"
					)
					returncode, pid, stderr = await asyncio.wait_for(
						tpu.execute_command(
							background_cmd,
							worker,
							stream=False,
							background=True,
						),
						timeout=timeout,
					)
					if returncode == 0:
						console.print(
							f"[green]Command started in background with PID: {pid}[/green]"
						)
						console.print("[green]Output will be saved to /tmp/nohup_*.out[/green]")
						config.save_command_history(command, "background", f"PID: {pid}")

						# Show how to check the process
						console.print("\n[yellow]To check process status:[/yellow]")
						console.print(f"eopod check-background {pid}")
						break
				else:
					returncode, stdout, stderr = await asyncio.wait_for(
						tpu.execute_command(
							command,
							worker,
							stream=stream,
							background=False,
						),
						timeout=timeout,
					)

					if returncode == 0:
						if not stream:
							progress.update(
								task,
								description="[green]Command completed successfully![/green]",
							)
							console.print("\nOutput:")
							console.print(stdout)
						else:
							console.print("[green]Command completed successfully![/green]")

						# Add command completion timestamp
						end_time = datetime.now()
						duration = end_time - start_time
						console.print(
							f"[cyan]Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]"
						)
						console.print(f"[cyan]Duration: {duration}[/cyan]")

						config.save_command_history(
							command,
							"success",
							stdout if not stream else "Streamed output",
						)
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
					task,
					description=f"[red]Error (attempt {attempt}):[/red] {str(e)}",
				)
				console.print(f"[red]Error (attempt {attempt}):[/red] {str(e)}")
				config.save_error_log(command, str(e))
				break

			if attempt < retry:
				progress.update(
					task,
					description=f"Retrying command in {delay} seconds... (Attempt {attempt + 1}/{retry})",
				)
				await asyncio.sleep(delay)
			else:
				progress.update(
					task,
					description=f"[red]Command failed after {retry} attempts[/red]",
				)


@cli.command()
@click.argument("pid_args", nargs=-1)
@click.option("--worker", default="all", help='Specific worker or "all"')
@async_command
async def check_background(pid_args, worker):
	"""Check status of background processes"""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)

	if pid_args:
		pids = " ".join(pid_args)
		command = f"ps -p {pids} -f"
	else:
		command = "ps aux | grep nohup"

	returncode, stdout, stderr = await tpu.execute_command(command, worker)

	if returncode == 0:
		console.print("[green]Background Processes:[/green]")
		console.print(stdout)
	else:
		console.print(f"[red]Error checking background processes:[/red] {stderr}")


# Add a command to kill background processes
@cli.command()
@click.argument("pid_args", nargs=-1, required=True)
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--force", is_flag=True, help="Force kill the process")
@async_command
async def kill(pid_args, worker, force):
	"""Kill a background process"""
	pids = " ".join(pid_args)
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)

	signal = "-9" if force else "-15"
	command = f"kill {signal} {pids}"

	returncode, stdout, stderr = await tpu.execute_command(command, worker)

	if returncode == 0:
		console.print(
			f"[green]Successfully {'force ' if force else ''}killed process(es) {pids}[/green]"
		)
	else:
		console.print(f"[red]Error killing process(es):[/red] {stderr}")


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
@click.option(
	"--worker",
	default="all",
	help='Specific worker or "all"',
)
@click.option(
	"--force",
	is_flag=True,
	help="Force kill all processes",
)
@click.option(
	"--pid",
	multiple=True,
	type=int,
	help="Specific PIDs to kill",
)
@async_command
async def kill_tpu(worker, force, pid):
	"""Kill processes using TPU resources"""
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)

	with Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
	) as progress:
		task = progress.add_task(description="Scanning for TPU processes...", total=None)

		try:
			# Get TPU status to determine number of workers
			status = await tpu.get_status()

			# Extract worker count from TPU status
			worker_count = 1  # Default to 1 for single TPU
			if "networkEndpoints" in status:
				worker_count = len(status["networkEndpoints"])

			workers = range(worker_count) if worker == "all" else [int(worker)]

			# Command to check if a process exists and is using TPU
			check_process_cmd = (
				"ps aux | grep -E 'python|jax|tensorflow' | "
				"grep -v grep | awk '{print $2}' | "
				"while read pid; do "
				"  if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps 2>/dev/null; then "
				"    echo $pid;"
				"  fi; "
				"done"
			)

			# Parallel process scanning
			async def scan_worker(w):
				returncode, stdout, stderr = await tpu.execute_command(
					check_process_cmd,
					worker=str(w),
					stream=False,
				)
				if returncode == 0 and stdout.strip():
					pids = [int(p.strip()) for p in stdout.splitlines() if p.strip()]
					return w, pids
				return w, []

			# Execute process scanning in parallel
			tasks = [scan_worker(w) for w in workers]
			results = await asyncio.gather(*tasks)

			worker_processes = {w: pids for w, pids in results if pids}

			if not worker_processes:
				console.print("[green]No TPU processes found.[/green]")
				return

			# Display found processes
			console.print("\n[yellow]Found TPU processes:[/yellow]")
			for w, pids in worker_processes.items():
				console.print(f"Worker {w}: PIDs {', '.join(map(str, pids))}")

			# If specific PIDs provided, filter them
			if pid:
				filtered_processes = {}
				for w, pids in worker_processes.items():
					matching_pids = [p for p in pids if p in pid]
					if matching_pids:
						filtered_processes[w] = matching_pids
				worker_processes = filtered_processes

			if not force:
				if not click.confirm("[yellow]Do you want to kill these processes?[/yellow]"):
					return

			# Parallel process killing
			async def kill_worker_processes(w, pids):
				results = []
				for pid in pids:
					kill_cmd = f"kill {'-9' if force else ''} {pid}"
					returncode, stdout, stderr = await tpu.execute_command(
						kill_cmd, worker=str(w), stream=False
					)
					results.append((pid, returncode == 0, stderr))
				return w, results

			# Execute process killing in parallel
			kill_tasks = [
				kill_worker_processes(w, pids) for w, pids in worker_processes.items()
			]
			kill_results = await asyncio.gather(*kill_tasks)

			# Process results
			for w, results in kill_results:
				for pid, success, error in results:
					if success:
						console.print(
							f"[green]Successfully killed process {pid} on worker {w}[/green]"
						)
					else:
						console.print(
							f"[red]Failed to kill process {pid} on worker {w}: {error}[/red]"
						)

			# Parallel cleanup
			cleanup_commands = [
				"sudo rm -f /tmp/libtpu_lockfile",
				"sudo rmmod tpu || true",
				"sudo modprobe tpu || true",
			]

			async def cleanup_worker(w):
				results = []
				for cmd in cleanup_commands:
					returncode, stdout, stderr = await tpu.execute_command(
						cmd, worker=str(w), stream=False
					)
					results.append((cmd, returncode == 0, stderr))
				return w, results

			# Execute cleanup in parallel
			cleanup_tasks = [cleanup_worker(w) for w in worker_processes.keys()]
			cleanup_results = await asyncio.gather(*cleanup_tasks)

			for w, results in cleanup_results:
				progress.update(task, description=f"Cleaned up TPU resources on worker {w}")

			# Verify TPU status
			progress.update(task, description="Verifying TPU status...")
			final_status = await tpu.get_status()
			console.print(
				f"[blue]Current TPU Status: {final_status.get('state', 'Unknown')}[/blue]"
			)

		except Exception as e:
			console.print(f"[red]Error during TPU process cleanup: {str(e)}[/red]")
			config.save_error_log("kill_tpu", str(e))


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


async def _smi_status(install_tpuinfo):
	config = EOConfig()
	project_id, zone, tpu_name = config.get_credentials()

	if not all([project_id, zone, tpu_name]):
		console.print("[red]Please configure EOpod first using 'eopod configure'[/red]")
		return

	tpu = TPUManager(project_id, zone, tpu_name)
	if install_tpuinfo:
		await tpu.execute_command("pip install tpu-info", stream=False)
	_, text, __ = await tpu.execute_command(
		'python -c "from tpu_info import cli;cli.print_chip_info()"',
		stream=False,
	)
	pattern = r"│\s+(\d+)\s+│\s+([\d.]+ GiB / [\d.]+ GiB)\s+│\s+([\d.]+%)\s+│"
	matches = re.findall(pattern, text)
	table_data = []
	for match in matches:
		device_index, memory_usage, duty_cycle = match
		table_data.append([int(device_index), memory_usage, duty_cycle])
	table_data_sorted = [
		[str(row[0]), row[1], row[2]] for row in sorted(table_data, key=lambda x: x[0])
	]
	table = Table(
		title="[bold magenta]TPU Utilization[/bold magenta]",
		title_justify="left",
	)
	# Add columns
	table.add_column("📟 Device Index", justify="center", style="bold blue")
	table.add_column("💾 Memory Usage", justify="left", style="white")
	table.add_column("⚡ Duty Cycle", justify="right", style="white")
	# Add rows to the table
	for row in table_data_sorted:
		table.add_row(str(row[0]), row[1], row[2])
	# Print the table
	console.print(table)


@cli.command()
@click.option(
	"--install-tpuinfo",
	is_flag=True,
	help="installs tpu-info (for first time only).",
)
@async_command
async def show_tpu_usage(install_tpuinfo):
	await _smi_status(install_tpuinfo)


@cli.command()
@click.option(
	"--install-tpuinfo",
	is_flag=True,
	help="installs tpu-info (for first time only).",
)
@async_command
async def smi(install_tpuinfo):
	await _smi_status(install_tpuinfo)


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
