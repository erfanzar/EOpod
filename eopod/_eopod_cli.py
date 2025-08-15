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
import logging
import pathlib
import re
import shlex
import subprocess
import time
from datetime import datetime

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from ._utils import EOPOD_PATH, PYTHON_PATH, EOConfig, TPUManager, async_command, run_command

console = Console(theme=Theme({"info": "cyan", "warning": "yellow", "error": "white", "success": "green"}))

logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


@click.group()
def cli():
    """eopod - Enhanced TPU Command Runner"""
    pass


def _get_config_and_manager():
    config = EOConfig()
    project_id, zone, tpu_name = config.get_credentials()
    if not all([project_id, zone, tpu_name]):
        console.print("[red]Please configure the tool first using 'eopod configure'[/red]")
        return

    tpu_manager = TPUManager(project_id, zone, tpu_name)
    return config, tpu_manager


@cli.command()
@click.option("--project-id", help="Google Cloud Project ID (optional if running on GCP)")
@click.option("--zone", help="Google Cloud Zone (optional if running on GCP)")
@click.option("--tpu-name", required=True, help="TPU Name")
def configure(project_id, zone, tpu_name):
    """Configure eopod with your Google Cloud details"""
    import re

    config = EOConfig()
    if "DEFAULT" not in config.config:
        config.config["DEFAULT"] = {}

    if not project_id:
        try:
            project_id = subprocess.check_output(
                "curl -s 'http://metadata.google.internal/computeMetadata/v1/project/project-id' -H 'Metadata-Flavor: Google'",  # noqa
                shell=True,
                text=True,
            ).strip()
            console.print(f"[yellow]Auto-detected project ID: {project_id}[/yellow]")
        except subprocess.CalledProcessError:
            console.print("[red]Failed to auto-detect project ID. Please provide it manually.[/red]")
            return

    if not zone:
        try:
            zone_output = subprocess.check_output(
                "curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/zone' -H 'Metadata-Flavor: Google'",  # noqa
                shell=True,
                text=True,
            ).strip()

            zone_match = re.search(r"/zones/([^/]+)", zone_output)
            if zone_match:
                zone = zone_match.group(1)
                console.print(f"[yellow]Auto-detected zone: {zone}[/yellow]")
            else:
                console.print("[red]Failed to parse auto-detected zone. Please provide it manually.[/red]")
                return
        except subprocess.CalledProcessError:
            console.print("[red]Failed to auto-detect zone. Please provide it manually.[/red]")
            return

    config.config["DEFAULT"]["project_id"] = project_id
    config.config["DEFAULT"]["zone"] = zone
    config.config["DEFAULT"]["tpu_name"] = tpu_name
    config.save_config()
    console.print("[green]Configuration saved successfully![/green]")


async def _install_package_uv(packages, uv_location):
    """
    Install one or more Python packages via uv on TPU workers.

    Example:
        install-package-uv torch numpy
    """
    _, tpu_manager = _get_config_and_manager()

    if uv_location is None:
        uv_location = str(pathlib.Path().home() / ".local" / "bin" / "uv")

    packages_str = " ".join(packages)

    cmd = f"{uv_location} pip install --python {PYTHON_PATH} {packages_str}"
    await tpu_manager.execute_command(cmd)


@cli.command()
@click.argument("packages", nargs=-1, required=True)
@click.option(
    "--uv-location",
    default=None,
    help="Path to uv executable (default: ~/.local/bin/uv)",
)
@async_command
async def install_package_uv(packages, uv_location):
    """
    Install one or more Python packages via uv on TPU workers.

    Example:
        install-package-uv torch numpy
    """
    await _install_package_uv(packages, uv_location)


@cli.command()
@async_command
async def get_internal_ips():
    """Get internal IP addresses of TPU workers."""
    config, tpu_manager = _get_config_and_manager()
    try:
        internal_ips = await tpu_manager.get_internal_ips()
        tpu_manager.display_ips(internal_ips, "internal", output_format="comma")
    except Exception as e:
        console.print(f"[red]Failed to get internal IPs: {e!s}[/red]")


@cli.command()
@async_command
async def get_external_ips():
    """Get external IP addresses of TPU workers."""

    config, tpu_manager = _get_config_and_manager()
    try:
        external_ips = await tpu_manager.get_external_ips()
        console.print(external_ips)
    except Exception as e:
        console.print(f"[red]Failed to get external IPs: {e!s}[/red]")


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("cmd_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--retry", default=1, help="Number of retries for failed commands")
@click.option("--delay", default=5, help="Delay between retries in seconds")
@click.option("--timeout", default=-1, help="Command timeout in seconds")
@click.option("--no-stream", is_flag=True, help="Disable output streaming")
@click.option("--background", is_flag=True, help="Run command in background")
@async_command
async def run(cmd_args, worker, retry, delay, timeout, no_stream, background):
    """Run a command on TPU VM with advanced features"""
    if not cmd_args:
        console.print("[red]No command provided[/red]")
        return

    command = " ".join(cmd_args)
    stream = not no_stream
    if timeout == -1:
        timeout = None

    config, tpu_manager = _get_config_and_manager()
    start_time = datetime.now()
    console.print(f"[cyan]Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")
    console.print(f"[cyan]Executing: {command}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        disable=stream,
    ) as progress:
        task = progress.add_task(description=f"Executing command: {command}", total=None)

        for attempt in range(1, retry + 1):
            try:
                returncode, stdout, stderr = await asyncio.wait_for(
                    tpu_manager.execute_command(command, worker, stream=stream, background=background),
                    timeout=timeout,
                )

                if returncode == 0:
                    if not stream and not background:
                        progress.update(task, description="[green]Command completed successfully![/green]")
                        console.print("\nOutput:")
                        console.print(stdout)

                    end_time = datetime.now()
                    duration = end_time - start_time
                    console.print(f"[cyan]Duration: {duration}[/cyan]")

                    config.save_command_history(command, "success", stdout if not stream else "Streamed output")
                    break
                else:
                    progress.update(task, description=f"[red]Attempt {attempt} failed[/red]")
                    console.print(f"[red]Error: {stderr}[/red]")
                    config.save_error_log(command, stderr)

            except TimeoutError:
                console.print(f"[red]Command timed out after {timeout} seconds[/red]")
                config.save_error_log(command, "Command timed out")
            except Exception as e:
                console.print(f"[red]Error: {e!s}[/red]")
                config.save_error_log(command, str(e))
                break

            if attempt < retry:
                await asyncio.sleep(delay)


@cli.command()
@click.argument("pid_args", nargs=-1)
@click.option("--worker", default="all", help='Specific worker or "all"')
@async_command
async def check_background(pid_args, worker):
    """Check status of background processes"""

    config, tpu = _get_config_and_manager()

    if pid_args:
        pids = " ".join(pid_args)
        command = f"ps -p {pids} -f"
    else:
        command = "ps aux | grep nohup | grep -v grep"

    returncode, stdout, stderr = await tpu.execute_command(command, worker)

    if returncode == 0:
        console.print("[green]Background Processes:[/green]")
        console.print(stdout)
    else:
        console.print(f"[red]Error checking background processes:[/red] {stderr}")


@cli.command()
@async_command
async def setup_path():
    """Add ~/.local/bin to PATH on all TPU workers if not already present"""
    config, tpu = _get_config_and_manager()

    path_command = """
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
        source ~/.bashrc
        echo "[success] Added ~/.local/bin to PATH"
    else
        echo "[info] ~/.local/bin is already in PATH"
    fi
    """

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task(description="Adding ~/.local/bin to PATH on all workers...", total=None)

        try:
            returncode, stdout, stderr = await tpu.execute_command(path_command, worker="all", stream=False)

            if returncode == 0:
                progress.update(task, description="[green]Successfully updated PATH on all workers[/green]")
                console.print("\nDetailed results:")
                console.print(stdout)
            else:
                progress.update(task, description=f"[red]Failed to update PATH: {stderr}[/red]")

        except Exception as e:
            progress.update(task, description=f"[red]Error: {e!s}[/red]")
            config.save_error_log("add_local_bin_to_path", str(e))


@cli.command()
@click.argument("pid_args", nargs=-1, required=True)
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--force", is_flag=True, help="Force kill the process")
@async_command
async def kill(pid_args, worker, force):
    """Kill a background process"""
    pids = " ".join(pid_args)
    config, tpu = _get_config_and_manager()

    signal = "-9" if force else "-15"
    command = f"kill {signal} {pids}"

    returncode, stdout, stderr = await tpu.execute_command(command, worker)

    if returncode == 0:
        console.print(f"[green]Successfully {'force ' if force else ''}killed process(es) {pids}[/green]")
    else:
        console.print(f"[red]Error killing process(es):[/red] {stderr}")


@cli.command()
@async_command
async def status():
    """Show TPU status and information"""
    config, tpu = _get_config_and_manager()
    try:
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
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--force", is_flag=True, help="Force kill all processes")
@click.option("--pid", multiple=True, type=int, help="Specific PIDs to kill")
@async_command
async def kill_tpu(worker, force, pid):
    """Kill processes using TPU resources"""
    config, tpu = _get_config_and_manager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task(description="Scanning for TPU processes...", total=None)

        try:
            status = await tpu.get_status()

            worker_count = 1
            if "networkEndpoints" in status:
                worker_count = len(status["networkEndpoints"])

            workers = range(worker_count) if worker == "all" else [int(worker)]

            check_process_cmd = (
                "ps aux | grep -E 'python|jax|tensorflow' | "
                "grep -v grep | awk '{print $2}' | "
                "while read pid; do "
                "  if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps 2>/dev/null; then "
                "    echo $pid;"
                "  fi; "
                "done"
            )

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

            tasks = [scan_worker(w) for w in workers]
            results = await asyncio.gather(*tasks)

            worker_processes = {w: pids for w, pids in results if pids}

            if not worker_processes:
                console.print("[green]No TPU processes found.[/green]")
                return

            console.print("\n[yellow]Found TPU processes:[/yellow]")
            for w, pids in worker_processes.items():
                console.print(f"Worker {w}: PIDs {', '.join(map(str, pids))}")

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

            async def kill_worker_processes(w, pids):
                results = []
                for pid in pids:
                    kill_cmd = f"kill {'-9' if force else ''} {pid}"
                    returncode, stdout, stderr = await tpu.execute_command(kill_cmd, worker=str(w), stream=False)
                    results.append((pid, returncode == 0, stderr))
                return w, results

            kill_tasks = [kill_worker_processes(w, pids) for w, pids in worker_processes.items()]
            kill_results = await asyncio.gather(*kill_tasks)

            for w, results in kill_results:
                for pid, success, error in results:
                    if success:
                        console.print(f"[green]Successfully killed process {pid} on worker {w}[/green]")
                    else:
                        console.print(f"[red]Failed to kill process {pid} on worker {w}: {error}[/red]")
            cleanup_commands = [
                "sudo rm -f /tmp/libtpu_lockfile",
                "sudo rmmod tpu || true",
                "sudo modprobe tpu || true",
            ]

            async def cleanup_worker(w):
                results = []
                for cmd in cleanup_commands:
                    returncode, stdout, stderr = await tpu.execute_command(cmd, worker=str(w), stream=False)
                    results.append((cmd, returncode == 0, stderr))
                return w, results

            cleanup_tasks = [cleanup_worker(w) for w in worker_processes.keys()]
            cleanup_results = await asyncio.gather(*cleanup_tasks)

            for w, results in cleanup_results:  # noqa
                progress.update(task, description=f"Cleaned up TPU resources on worker {w}")

            progress.update(task, description="Verifying TPU status...")
            final_status = await tpu.get_status()
            console.print(f"[blue]Current TPU Status: {final_status.get('state', 'Unknown')}[/blue]")

        except Exception as e:
            console.print(f"[red]Error during TPU process cleanup: {e!s}[/red]")
            config.save_error_log("kill_tpu", str(e))


async def _execute_terminal_command(project_id, zone, tpu_name, command, worker):
    """Helper function to execute commands in terminal mode"""
    try:
        tpu = TPUManager(project_id, zone, tpu_name)

        # Show a simple spinner while executing
        with Progress(SpinnerColumn(), TextColumn("Executing..."), console=console) as progress:
            task = progress.add_task("exec", total=None)
            returncode, stdout, stderr = await tpu.execute_command(command, worker, stream=False)
            progress.remove_task(task)

        if returncode == 0:
            if stdout.strip():
                console.print(stdout)
        else:
            console.print(f"[red]Command failed (exit code {returncode})[/red]")
            if stderr.strip():
                console.print(f"[red]{stderr}[/red]")

    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")


async def _show_status_async(project_id, zone, tpu_name):
    """Helper function to show TPU status in terminal"""
    try:
        tpu = TPUManager(project_id, zone, tpu_name)
        status = await tpu.get_status()

        table = Table(title="TPU Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", status.get("name", ""))
        table.add_row("State", status.get("state", ""))
        table.add_row("Type", status.get("acceleratorType", ""))
        table.add_row("Network", status.get("network", ""))

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error fetching status: {e}[/red]")


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
        table.add_row(entry["timestamp"], entry["command"], entry["status"], entry["output"])

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
        console.print("[red]No configuration found. Please run 'eopod configure' first.[/red]")


@cli.command()
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--shell", default="/bin/bash", help="Shell to use (default: /bin/bash)")
def terminal(worker, shell):
    """Open an interactive terminal session with TPU workers"""
    config = EOConfig()
    project_id, zone, tpu_name = config.get_credentials()

    if not all([project_id, zone, tpu_name]):
        console.print("[red]Please configure eopod first using 'eopod configure'[/red]")
        return

    # Show welcome message
    welcome_panel = Panel.fit(
        f"[bold green]TPU Interactive Terminal[/bold green]\n"
        f"[cyan]TPU:[/cyan] {tpu_name}\n"
        f"[cyan]Worker:[/cyan] {worker}\n"
        f"[cyan]Zone:[/cyan] {zone}\n\n"
        f"[yellow]Commands:[/yellow]\n"
        f"  [bold]exit[/bold] or [bold]quit[/bold] - Exit terminal\n"
        f"  [bold]:help[/bold] - Show help\n"
        f"  [bold]:status[/bold] - Show TPU status\n"
        f"  [bold]:worker <num>[/bold] - Switch to specific worker\n"
        f"  [bold]:background <cmd>[/bold] - Run command in background\n",
        title="Welcome",
        border_style="blue",
    )
    console.print(welcome_panel)

    current_worker = worker

    while True:
        try:
            # Create a rich prompt
            prompt_text = (
                f"[bold green]eopod[/bold green]:[bold blue]{tpu_name}[/bold blue]:"
                f"[bold yellow]worker-{current_worker}[/bold yellow]$ "
            )
            command = Prompt.ask(prompt_text, console=console)

            if not command.strip():
                continue

            # Handle special commands
            if command.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif command.startswith(":help"):
                help_panel = Panel.fit(
                    "[bold yellow]Available Commands:[/bold yellow]\n\n"
                    "[bold]Regular commands[/bold] - Execute directly on TPU\n"
                    "[bold]:help[/bold] - Show this help\n"
                    "[bold]:status[/bold] - Show current TPU status\n"
                    "[bold]:worker <num>[/bold] - Switch to specific worker (or 'all')\n"
                    "[bold]:background <cmd>[/bold] - Run command in background\n"
                    "[bold]:history[/bold] - Show recent command history\n"
                    "[bold]:clear[/bold] - Clear screen\n"
                    "[bold]exit/quit[/bold] - Exit terminal\n",
                    title="Help",
                    border_style="yellow",
                )
                console.print(help_panel)
            elif command.startswith(":status"):
                # Run async status command
                asyncio.run(_show_status_async(project_id, zone, tpu_name))
            elif command.startswith(":worker"):
                parts = command.split()
                if len(parts) == 2:
                    new_worker = parts[1]
                    current_worker = new_worker
                    console.print(f"[green]Switched to worker: {current_worker}[/green]")
                else:
                    console.print("[red]Usage: :worker <worker_num|all>[/red]")
            elif command.startswith(":background"):
                bg_command = command[11:].strip()  # Remove ':background '
                if bg_command:
                    console.print(f"[yellow]Running in background: {bg_command}[/yellow]")
                    asyncio.run(_execute_background_command(project_id, zone, tpu_name, bg_command, current_worker))
                else:
                    console.print("[red]Usage: :background <command>[/red]")
            elif command.startswith(":history"):
                _show_history()
            elif command.startswith(":clear"):
                console.clear()
            else:
                # Execute regular command on TPU
                console.print(f"[cyan]Executing on worker {current_worker}: {command}[/cyan]")
                asyncio.run(_execute_terminal_command(project_id, zone, tpu_name, command, current_worker))

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' or 'quit' to leave the terminal[/yellow]")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def _execute_background_command(project_id, zone, tpu_name, command, worker):
    """Helper function to execute background commands"""
    try:
        tpu = TPUManager(project_id, zone, tpu_name)
        returncode, pid, stderr = await tpu.execute_command(command, worker, background=True)

        if returncode == 0:
            console.print(f"[green]Background process started with PID: {pid}[/green]")
        else:
            console.print(f"[red]Failed to start background process: {stderr}[/red]")

    except Exception as e:
        console.print(f"[red]Error starting background process: {e}[/red]")


def _show_history():
    """Helper function to show command history in terminal"""
    config = EOConfig()

    if not config.history_file.exists():
        console.print("[yellow]No command history found.[/yellow]")
        return

    with open(config.history_file, "r") as f:
        history = yaml.safe_load(f) or []

    if not history:
        console.print("[yellow]No command history found.[/yellow]")
        return

    table = Table(title="Recent Command History")
    table.add_column("Time", style="cyan")
    table.add_column("Command", style="white")
    table.add_column("Status", style="green")

    # Show last 10 commands
    for entry in history[-10:]:
        timestamp = entry["timestamp"].split("T")[1][:8]  # Show only time part
        table.add_row(timestamp, entry["command"][:50], entry["status"])

    console.print(table)


@cli.command()
@async_command
async def smi():
    """Show TPU utilization (like nvidia-smi)"""
    config, tpu = _get_config_and_manager()

    try:
        _, text, _ = await tpu.execute_command(
            f'{PYTHON_PATH} -c "from tpu_info import cli; cli.print_chip_info()"',
            stream=False,
        )

        pattern = r"│\s+(\d+)\s+│\s+([\d.]+ GiB / [\d.]+ GiB)\s+│\s+([\d.]+%)\s+│"
        matches = re.findall(pattern, text)

        if matches:
            table = Table(title="[bold magenta]TPU Utilization[/bold magenta]")
            table.add_column("📟 Device", justify="center", style="bold blue")
            table.add_column("💾 Memory Usage", justify="left", style="white")
            table.add_column("⚡ Duty Cycle", justify="right", style="white")

            for device_index, memory_usage, duty_cycle in matches:
                table.add_row(device_index, memory_usage, duty_cycle)

            console.print(table)
        else:
            console.print("[yellow]Could not parse TPU utilization data[/yellow]")
            console.print(text)  # Show raw output

    except Exception as e:
        console.print(f"[red]Error getting TPU utilization: {e}[/red]")


@cli.command()
@async_command
async def clean_logs():
    """Clean up logs and temporary files on the TPU VM"""
    config, tpu = _get_config_and_manager()
    command = """
    sudo bash -c 'echo "[*] Vacuuming journal logs (keeping 1 second)..." && journalctl --vacuum-time=1s && echo "[*] Deleting rotated/compressed logs..." && find /var/log -type f \( -name "*.gz" -o -name "*.1" -o -name "*.old" -o -name "*.bak" -o -name "*-????????" -o -name "*.log.[0-9]*" \) -print -delete && echo "[*] Truncating active log files..." && find /var/log -type f -name "*.log" -exec truncate -s 0 {} \; && echo "[*] Vacuuming journal logs to 50MB cap..." && journalctl --vacuum-size=5M && docker system prune -af --volumes && echo "[✔] Cleanup complete."'
    """  # noqa
    await tpu.execute_command(command.strip(), stream=False)


@cli.command()
@click.option("--external", is_flag=True, help="Use external IPs instead of internal IPs")
@click.option("--stop", is_flag=True, help="Stop the Ray cluster")
@click.option("--verify", is_flag=True, help="Verify the Ray cluster setup")
@click.option("--tpu-version", help="Set TPU version (auto-detected if not provided)")
@click.option("--tpu-slice", type=int, help="Set TPU slice size (auto-detected if not provided)")
@click.option("--num-slices", type=int, default=1, help="Number of TPU slices to combine (default: 1)")
@click.option("--ssh-user", help="SSH username to use")
@click.option("--config", help="Path to YAML config file with IP addresses")
@click.option("--test-ssh", is_flag=True, help="Test SSH connectivity to all nodes")
@click.option("--external-ips", help="Comma-separated list of external IPs")
@click.option("--self-job", is_flag=True, help="Run only on the current machine (no SSH)")
@click.option("--slice-config", help="Path to YAML config file with slice configurations")
@click.option("--python-path", help="Path to venv or python interpreter")
@click.option("--head-node-ip", help="IP address of external head node (if not using first IP in list)")
@click.option("--head-only", is_flag=True, help="Run this node as head only (no TPU resources)")
@click.option("--spot-tpu-name", help="Name of spot TPU to configure as workers")
@click.option("--spot-tpu-project-id", help="Project ID for spot TPU (defaults to current)")
@click.option("--spot-tpu-zone", help="Zone for spot TPU (defaults to current)")
@click.option("--head-ip", help="IP address to use as head (defaults to current machine)")
def auto_config_ray(
    external,
    stop,
    verify,
    tpu_version,
    tpu_slice,
    num_slices,
    ssh_user,
    config,
    test_ssh,
    external_ips,
    self_job,
    slice_config,
    python_path,
    head_node_ip,
    head_only,
    spot_tpu_name,
    spot_tpu_project_id,
    spot_tpu_zone,
    head_ip,
):
    """
    Auto-configure Ray on TPU cluster using internal IPs from current setup.
    Automatically detects TPU version and slice size if not specified.

    Examples:
        # Setup v4-64 with external v2-8 head:
        eformer auto-config-ray --self-job --head-node-ip 10.x.x.x

        # Setup v2-8 as head-only:
        eformer auto-config-ray --self-job --head-only --head-node-ip 10.x.x.x
    """
    import asyncio
    import re
    import subprocess

    try:
        current_internal_ip = subprocess.check_output("hostname -I", shell=True, text=True).strip().split()[0]
        current_external_ip = subprocess.check_output("curl -s https://api.ipify.org", shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        console.print("[red]Could not determine current machine's IP[/red]")
        return

    if not head_ip:
        head_ip = current_external_ip if external else current_internal_ip
        console.print(f"[green]Using current machine as head: {head_ip}[/green]")

    is_head_machine = head_ip in [current_internal_ip, current_external_ip]
    if spot_tpu_name and (not spot_tpu_project_id or not spot_tpu_zone):
        try:
            if not spot_tpu_project_id:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                spot_tpu_project_id = result.stdout.strip()
                console.print(f"[green]Using current project for spot TPU: {spot_tpu_project_id}[/green]")

            if not spot_tpu_zone:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "compute/zone"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                spot_tpu_zone = result.stdout.strip()
                console.print(f"[green]Using current zone for spot TPU: {spot_tpu_zone}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to get default project/zone: {e!s}[/red]")
            return

    if spot_tpu_name:
        console.print(f"[cyan]Configuring spot TPU '{spot_tpu_name}' to connect to head at {head_ip}[/cyan]")

        async def fetch_spot_tpu_ips():
            console.print(f"[yellow]Fetching IPs for spot TPU: {spot_tpu_name}[/yellow]")
            manager = TPUManager(spot_tpu_project_id, spot_tpu_zone, spot_tpu_name)

            try:
                internal_ips_dict = await manager.get_internal_ips()
                internal_ips_list = list(internal_ips_dict.values())

                external_ips_list = []
                if external:
                    external_ips_str = await manager.get_external_ips()
                    external_ips_list = [ip.strip() for ip in external_ips_str.split(",") if ip.strip()]

                return internal_ips_list, external_ips_list
            except Exception as e:
                console.print(f"[red]Failed to fetch IPs for {spot_tpu_name}: {e!s}[/red]")
                return [], []

        spot_internal_ips, spot_external_ips = asyncio.run(fetch_spot_tpu_ips())

        if not spot_internal_ips:
            console.print("[red]No IPs found for spot TPU[/red]")
            return

        if is_head_machine and not stop:
            console.print("[cyan]Setting up current machine as Ray head...[/cyan]")

            try:
                accelerator_type = subprocess.check_output(
                    "curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type'"
                    " -H 'Metadata-Flavor: Google'",
                    shell=True,
                    text=True,
                ).strip()

                match = re.match(r"v(\d+[a-zA-Z]?)-(\d+)", accelerator_type)
                if match:
                    head_tpu_version = f"v{match.group(1)}"
                    head_tpu_slice = int(match.group(2))
                    head_has_tpu = True
                    console.print(f"[green]Head has TPU: {head_tpu_version}-{head_tpu_slice}[/green]")
                else:
                    head_has_tpu = False
            except Exception:
                head_has_tpu = False
                console.print("[yellow]Head machine has no TPU (CPU-only head)[/yellow]")

            cmd = [
                python_path or PYTHON_PATH,
                "-m",
                "eformer.executor.tpu_patch_ray",
                "--self-job",
                "--head-only" if not head_has_tpu else "",
                "--head-node-ip",
                head_ip,
            ].filter(None)

            subprocess.run(" ".join(cmd), shell=True, check=True)
            console.print("[green]Ray head started successfully[/green]")
            time.sleep(5)

        console.print(f"[cyan]Configuring {len(spot_internal_ips)} spot TPU workers...[/cyan]")

        cmd_parts = [
            python_path or PYTHON_PATH,
            "-m",
            "eformer.executor.tpu_patch_ray",
            "--tpu-version",
            tpu_version or "v4",
            "--tpu-slice",
            str(tpu_slice or len(spot_internal_ips) * 8),
            "--internal-ips",
            ",".join(spot_internal_ips),
            "--self-job",
            "--head-node-ip",
            head_ip,
        ]

        if external and spot_external_ips:
            cmd_parts.extend(["--external-ips", ",".join(spot_external_ips)])

        if stop:
            cmd_parts.append("--stop")

        gcloud_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            spot_tpu_name,
            f"--zone={spot_tpu_zone}",
            f"--project={spot_tpu_project_id}",
            "--worker=all",
            "--command",
            shlex.quote(" ".join(shlex.quote(str(part)) for part in cmd_parts)),
        ]

        final_cmd = " ".join(gcloud_cmd)
        console.print(f"[yellow]Executing on spot TPU: {final_cmd}[/yellow]")

        try:
            subprocess.run(final_cmd, shell=True, check=True)
            console.print("[green]Spot TPU workers configured successfully![/green]")

            if not stop:
                console.print("\n[cyan]Ray cluster ready![/cyan]")
                console.print(f"[cyan]Head node: {head_ip}[/cyan]")
                console.print(f"[cyan]Workers: {len(spot_internal_ips)} nodes from {spot_tpu_name}[/cyan]")
                console.print(f"[cyan]Dashboard: http://{head_ip}:8265[/cyan]")
                console.print(f"[cyan]Connect with: ray.init(address='{head_ip}:6379')[/cyan]")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to configure spot TPU workers: {e!s}[/red]")
    else:
        if head_only:
            console.print("[cyan]Running in head-only mode (no TPU resources)[/cyan]")
            cmd_parts = [
                EOPOD_PATH,
                "run",
                python_path or PYTHON_PATH,
                "-m",
                "eformer.executor.tpu_patch_ray",
                "--head-only",
                "--self-job",
            ]

            if head_node_ip:
                cmd_parts.extend(["--head-node-ip", head_node_ip])
            else:
                # Get current machine's IP for head-only mode
                try:
                    local_ip = subprocess.check_output("hostname -I", shell=True, text=True).strip().split()[0]
                    cmd_parts.extend(["--head-node-ip", local_ip])
                    console.print(f"[green]Using local IP as head node: {local_ip}[/green]")
                except subprocess.CalledProcessError:
                    console.print("[red]Could not determine local IP for head node[/red]")
                    return

            if stop:
                cmd_parts.append("--stop")
            if verify:
                cmd_parts.append("--verify")

            final_cmd = " ".join(shlex.quote(str(part)) for part in cmd_parts)
            console.print(f"[yellow]Executing: {final_cmd}[/yellow]")

            try:
                subprocess.run(final_cmd, shell=True, check=True, text=True)
                console.print("[green]Head-only node configuration completed successfully![/green]")
                if not stop:
                    console.print("[cyan]Now run the worker setup on your TPU nodes with --head-node-ip[/cyan]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to configure head-only node: {e!s}[/red]")
            return

        # Regular TPU node setup (existing logic)
        try:
            console.print("[yellow]Fetching internal IPs from eopod...[/yellow]")
            internal_ips_output = subprocess.check_output(
                f"{EOPOD_PATH} get-internal-ips", shell=True, text=True
            ).strip()

            sanitized_ips_output = internal_ips_output.replace("\n", "").replace("\r", "")
            internal_ips = [ip.strip() for ip in sanitized_ips_output.split(",") if ip.strip()]
            if not internal_ips:
                console.print("[red]No internal IPs found. Make sure eopod is configured correctly.[/red]")
                return

            internal_ips_str = ",".join(internal_ips)
            console.print(f"[green]Found internal IPs: {internal_ips_str}[/green]")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to get internal IPs: {e!s}[/red]")
            return

        if not tpu_version or not tpu_slice:
            try:
                console.print("[yellow]Auto-detecting TPU configuration...[/yellow]")
                accelerator_type = subprocess.check_output(
                    "curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type' -H 'Metadata-Flavor: Google'",  # noqa
                    shell=True,
                    text=True,
                ).strip()

                match = re.match(r"v(\d+[a-zA-Z]?)-(\d+)", accelerator_type)
                if match:
                    detected_version = match.group(1)
                    detected_slice = int(match.group(2))

                    if not tpu_version:
                        tpu_version = f"v{detected_version}"
                        console.print(f"[green]Auto-detected TPU version: {tpu_version}[/green]")

                    if not tpu_slice:
                        tpu_slice = detected_slice
                        console.print(f"[green]Auto-detected TPU slice size: {tpu_slice}[/green]")
                else:
                    console.print(
                        f"[yellow]Could not parse accelerator type: {accelerator_type}. Please provide --tpu-version and --tpu-slice manually.[/yellow]"  # noqa
                    )
                    if not tpu_version or not tpu_slice:
                        console.print("[red]TPU version and slice size are required. Exiting.[/red]")
                        return
            except subprocess.CalledProcessError:
                console.print(
                    "[yellow]Failed to auto-detect TPU configuration. Please provide --tpu-version and --tpu-slice manually.[/yellow]"  # noqa
                )
                if not tpu_version or not tpu_slice:
                    console.print("[red]TPU version and slice size are required. Exiting.[/red]")
                    return

        # Show configuration summary
        if head_node_ip:
            console.print(f"[cyan]Using external head node at: {head_node_ip}[/cyan]")
            console.print("[cyan]This TPU cluster will connect as workers to the external head[/cyan]")

        cmd_parts = [
            EOPOD_PATH,
            "run",
            python_path or PYTHON_PATH,
            "-m",
            "eformer.executor.tpu_patch_ray",
        ]

        cmd_parts.extend(["--tpu-version", str(tpu_version)])
        cmd_parts.extend(["--tpu-slice", str(tpu_slice)])
        cmd_parts.extend(["--num-slices", str(num_slices)])
        cmd_parts.extend(["--internal-ips", internal_ips_str])

        if external:
            cmd_parts.append("--external")
        if stop:
            cmd_parts.append("--stop")
        if verify:
            cmd_parts.append("--verify")
        if self_job:
            cmd_parts.append("--self-job")
        if test_ssh:
            cmd_parts.append("--test-ssh")

        if ssh_user:
            cmd_parts.extend(["--ssh-user", ssh_user])
        if config:
            cmd_parts.extend(["--config", config])
        if external_ips:
            cmd_parts.extend(["--external-ips", external_ips])
        if slice_config:
            cmd_parts.extend(["--slice-config", slice_config])
        if head_node_ip:
            cmd_parts.extend(["--head-node-ip", head_node_ip])

        final_cmd = " ".join(shlex.quote(str(part)) for part in cmd_parts)
        console.print(f"[yellow]Executing: {final_cmd}[/yellow]")

        try:
            subprocess.run(final_cmd, shell=True, check=True, text=True)
            console.print("[green]Ray cluster configuration completed successfully![/green]")

            if head_node_ip and not stop:
                console.print(f"\n[cyan]Workers connected to head node at: {head_node_ip}[/cyan]")
                console.print(f"[cyan]Ray dashboard available at: http://{head_node_ip}:8265[/cyan]")
                console.print(f"[cyan]Connect to cluster with: ray.init(address='{head_node_ip}:6379')[/cyan]")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to configure Ray cluster: {e!s}[/red]")


@cli.command()
@click.option(
    "--port",
    "-p",
    type=int,
    required=True,
    multiple=True,
    help="Port number(s) to open. Can be specified multiple times (e.g., -p 80 -p 443).",
)
@click.option(
    "--direction",
    type=click.Choice(["ingress", "egress", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Direction of traffic to allow.",
)
@click.option(
    "--protocol",
    default="tcp",
    show_default=True,
    type=click.Choice(["tcp", "udp", "icmp", "all"], case_sensitive=False),
    help="Protocol to allow.",
)
@click.option(
    "--target-tag",
    default=None,
    help="Network tag for VMs. If omitted, defaults to 'tpu-<your-tpu-name>'. IMPORTANT: VMs must have this tag!",
)
@click.option(
    "--source-ranges",
    default="0.0.0.0/0",
    show_default=True,
    help="Source IP CIDR range for ingress rules.",
)
@click.option(
    "--destination-ranges",
    default="0.0.0.0/0",
    show_default=True,
    help="Destination IP CIDR range for egress rules.",
)
@click.option(
    "--priority",
    type=int,
    default=1000,
    show_default=True,
    help="Firewall rule priority (lower number = higher priority).",
)
@click.option(
    "--description",
    default="Rule created by eopod",
    show_default=True,
    help="Description for the firewall rule.",
)
@click.option(
    "--network",
    default=None,
    help="Network name to use. If omitted, will attempt to detect from TPU configuration.",
)
@click.option(
    "--update-existing/--skip-existing",
    default=False,
    show_default=True,
    help="Whether to update existing rules or skip them.",
)
@click.option(
    "--verify-tag",
    is_flag=True,
    default=False,
    help="Verify that the target tag is applied to the TPU VM before creating rules.",
)
@async_command
async def open_port(
    port,
    direction,
    protocol,
    target_tag,
    source_ranges,
    destination_ranges,
    priority,
    description,
    network,
    update_existing,
    verify_tag,
):
    """Creates GCP firewall rules to open ports for TPU VMs."""
    config = EOConfig()
    project_id, zone, tpu_name = config.get_credentials()

    if not all([project_id, zone, tpu_name]):
        console.print("[red]Please configure eopod first using 'eopod configure'[/red]")
        return

    safe_tpu_name = tpu_name.lower().replace("_", "-")

    effective_target_tag = target_tag if target_tag is not None else f"{safe_tpu_name}"

    tpu_manager = TPUManager(project_id, zone, tpu_name)

    if network is None:
        try:
            tpu_info = await tpu_manager.get_tpu_info()
            network = tpu_info.get("networkConfig", {}).get("network", "default")
            console.print(f"Using network: {network}")
        except Exception as e:
            console.print(f"[yellow]Could not determine network from TPU config: {e}[/yellow]")
            console.print("[yellow]Using 'default' network instead[/yellow]")
            network = "default"

    if verify_tag:
        try:
            tpu_info = await tpu_manager.get_tpu_info()
            vm_tags = tpu_info.get("networkConfig", {}).get("networkTags", [])
            if effective_target_tag not in vm_tags:
                console.print(f"[red]Target tag '{effective_target_tag}' is not applied to the TPU VM![/red]")
                console.print(f"[yellow]Available tags: {', '.join(vm_tags) if vm_tags else 'None'}[/yellow]")
                if click.confirm("Do you want to add this tag to the TPU VM?", default=False):
                    console.print("[yellow]Adding tag functionality not implemented yet[/yellow]")
                else:
                    return
        except Exception as e:
            console.print(f"[yellow]Could not verify tags: {e}[/yellow]")
            if not click.confirm("Continue without verifying tags?", default=False):
                return

    directions_to_process = ["ingress", "egress"] if direction.lower() == "both" else [direction.lower()]

    for p in port:
        for current_direction in directions_to_process:
            rule_name = f"a-allow-{safe_tpu_name}-{p}-{current_direction}".lower()[:63]

            try:
                cmd = f"gcloud compute firewall-rules describe {rule_name} --project={project_id} --format=json"
                _ = await run_command(cmd, capture_output=True)
                rule_exists = True
                console.print(f"Rule '{rule_name}' already exists.")

                if not update_existing:
                    console.print("[yellow]Skipping (use --update-existing to update)[/yellow]")
                    continue

            except Exception:
                rule_exists = False

            cmd_parts = [
                "gcloud",
                "compute",
                "firewall-rules",
                "update" if rule_exists else "create",
                rule_name,
                f"--project={project_id}",
                f"--direction={current_direction.upper()}",
                f"--priority={priority}",
                f"--network={network}",
                "--action=ALLOW",
            ]

            if protocol == "all":
                cmd_parts.append("--rules=all")
            elif protocol == "icmp":
                cmd_parts.append("--rules=icmp")
            else:
                cmd_parts.append(f"--rules={protocol}:{p}")

            if current_direction.lower() == "ingress":
                cmd_parts.append(f"--source-ranges={source_ranges}")

            if current_direction.lower() == "egress":
                cmd_parts.append(f"--destination-ranges={destination_ranges}")

            cmd_parts.append(f"--target-tags={effective_target_tag}")

            cmd_parts.append(f"--description='{description}'")

            cmd = " ".join(cmd_parts)
            console.print(f"[green]Executing:[/green]\n{cmd}")

            try:
                await run_command(cmd)
                console.print(
                    f"[green]Successfully {'updated' if rule_exists else 'created'} firewall rule '{rule_name}'[/green]"
                )
            except Exception as e:
                console.print(f"[red]Failed to {'update' if rule_exists else 'create'} firewall rule: {e}[/red]")


@cli.command()
def errors():
    """Show recent command execution errors"""
    config = EOConfig()

    if not config.error_log_file.exists():
        console.print("[yellow]No error log found.[/yellow]")
        return

    with open(config.error_log_file, "r") as f:
        try:
            error_log = yaml.safe_load(f) or []
        except yaml.YAMLError as e:
            console.print(f"[red]Error loading error log: {e}[/red]")
            return

    if not error_log:
        console.print("[green]No errors found![/green]")
        return

    table = Table(title="Error Log", style="red")
    table.add_column("Timestamp")
    table.add_column("Command")
    table.add_column("Error")

    for entry in error_log[-10:]:  # Show last 10 errors
        table.add_row(entry["timestamp"], entry["command"][:30], entry["error"][:100])

    console.print(table)


def main():
    """
    Main entry point for the eopod CLI.
    """
    try:
        asyncio.run(cli())
    except click.exceptions.Exit as e:
        if e.exit_code != 0:
            console.print(f"[red]Error:[/red] Command failed with exit code {e.exit_code}")
            logging.exception("Click command failed")
    except Exception as e:
        console.print(f"[red]Unexpected Error:[/red] {e!s}")
