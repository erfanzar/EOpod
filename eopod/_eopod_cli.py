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
import json
import logging
import pathlib
import re
import shlex
import shutil
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


def _read_gcloud_config_property(property_name: str) -> str:
    """Read gcloud config property and fail clearly when unset."""
    result = subprocess.run(
        ["gcloud", "config", "get", property_name],
        capture_output=True,
        text=True,
        check=True,
    )
    value = result.stdout.strip()
    if not value or value == "(unset)":
        raise ValueError(f"gcloud config property '{property_name}' is unset")
    return value


def _network_name(network_value: str | None) -> str:
    """Normalize full network URI to simple network name."""
    if not network_value:
        return "default"
    return network_value.rsplit("/", 1)[-1]


def _basename(resource_name: str | None) -> str | None:
    if not resource_name:
        return None
    return resource_name.rsplit("/", 1)[-1]


def _metadata_value(path: str) -> str | None:
    url = f"http://metadata.google.internal/computeMetadata/v1/{path}"
    result = subprocess.run(
        ["curl", "-fsS", "-H", "Metadata-Flavor: Google", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _detect_project_id_from_metadata() -> str | None:
    return _metadata_value("project/project-id")


def _detect_zone_from_metadata() -> str | None:
    zone_output = _metadata_value("instance/zone")
    if not zone_output:
        return None
    zone_match = re.search(r"/zones/([^/]+)", zone_output)
    if zone_match:
        return zone_match.group(1)
    return None


def _detect_self_internal_ip_from_metadata() -> str | None:
    return _metadata_value("instance/network-interfaces/0/ip")


def _list_tpu_vms(project_id: str, zone: str) -> list[dict]:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        f"--project={project_id}",
        f"--zone={zone}",
        "--format=json(name,queuedResource,state,networkEndpoints.ipAddress)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to list TPU VMs")
    try:
        data = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse TPU VM list output: {e}") from e
    return data if isinstance(data, list) else []


def _detect_tpu_identity_from_current_vm(project_id: str, zone: str) -> tuple[str | None, str | None]:
    self_ip = _detect_self_internal_ip_from_metadata()
    if not self_ip:
        return None, None

    # Preferred path: let gcloud do endpoint flatten/filter directly.
    try:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={project_id}",
            f"--zone={zone}",
            "--flatten=networkEndpoints[]",
            f"--filter=networkEndpoints.ipAddress={self_ip}",
            "--format=value(name.basename(),queuedResource.basename())",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            line = next((ln.strip() for ln in result.stdout.splitlines() if ln.strip()), "")
            if line:
                parts = line.split()
                detected_tpu = parts[0] if len(parts) >= 1 else None
                detected_queued = parts[1] if len(parts) >= 2 else None
                return detected_tpu, detected_queued
    except Exception:
        pass

    # Fallback path: parse structured JSON from list output.
    try:
        for node in _list_tpu_vms(project_id, zone):
            endpoints = node.get("networkEndpoints", []) or []
            endpoint_ips = [ep.get("ipAddress") for ep in endpoints if ep.get("ipAddress")]
            if self_ip in endpoint_ips:
                return _basename(node.get("name")), _basename(node.get("queuedResource"))
    except Exception:
        return None, None
    return None, None


def _resolve_tpu_from_queued_resource(project_id: str, zone: str, queued_resource: str) -> str | None:
    queued_resource = _basename(queued_resource) or queued_resource
    try:
        matches = []
        for node in _list_tpu_vms(project_id, zone):
            queued_basename = _basename(node.get("queuedResource"))
            if queued_basename == queued_resource:
                matches.append(node)

        if not matches:
            return None

        active = next((m for m in matches if str(m.get("state", "")).upper() in {"READY", "ACTIVE"}), None)
        selected = active or matches[0]
        return _basename(selected.get("name"))
    except Exception:
        return None


def _describe_tpu_vm(project_id: str, zone: str, tpu_name: str) -> dict | None:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        tpu_name,
        f"--project={project_id}",
        f"--zone={zone}",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None


def _normalize_target_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = []
    seen = set()
    for tag in tags or []:
        value = str(tag).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _preferred_tpu_target_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = _normalize_target_tags(tags)
    if not normalized:
        return []

    # TPU-specific per-node tags are the safest match for firewall rules.
    tpu_specific = [tag for tag in normalized if tag.startswith("tpu-")]
    if tpu_specific:
        return tpu_specific

    non_google = [tag for tag in normalized if not tag.startswith("x-google-")]
    return non_google or normalized


def _detect_instance_tags_from_metadata() -> list[str]:
    raw_tags = _metadata_value("instance/tags")
    if not raw_tags:
        return []

    try:
        parsed = json.loads(raw_tags)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return _preferred_tpu_target_tags(parsed)

    return _preferred_tpu_target_tags([line.strip() for line in raw_tags.splitlines() if line.strip()])


def _project_number_from_resource_name(resource_name: str | None) -> str | None:
    if not resource_name:
        return None

    match = re.search(r"(?:^|/)projects/(\d+)(?:/|$)", resource_name)
    if match:
        return match.group(1)
    return None


def _read_project_number(project_id: str) -> str | None:
    result = subprocess.run(
        ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _describe_firewall_rule(project_id: str, rule_name: str) -> dict | None:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "firewall-rules",
            "describe",
            rule_name,
            f"--project={project_id}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None


def _resolve_tpu_vm_target_tags(
    project_id: str,
    zone: str,
    tpu_name: str,
    tpu_info: dict | None = None,
) -> list[str]:
    tpu_info = tpu_info or _describe_tpu_vm(project_id, zone, tpu_name)
    if not tpu_info:
        return []

    direct_tags = _preferred_tpu_target_tags(
        tpu_info.get("tags") or tpu_info.get("networkConfig", {}).get("networkTags", [])
    )
    if direct_tags:
        return direct_tags

    project_number = (
        _project_number_from_resource_name(tpu_info.get("queuedResource"))
        or _project_number_from_resource_name(tpu_info.get("name"))
        or _read_project_number(project_id)
    )
    tpu_id = str(tpu_info.get("id") or "").strip()
    if project_number and tpu_id:
        healthcheck_rule = _describe_firewall_rule(
            project_id,
            f"tpu-{project_number}-{tpu_id}-healthcheck-fw",
        )
        healthcheck_tags = _preferred_tpu_target_tags((healthcheck_rule or {}).get("targetTags", []))
        if healthcheck_tags:
            return healthcheck_tags

    detected_tpu, _ = _detect_tpu_identity_from_current_vm(project_id, zone)
    if detected_tpu == tpu_name:
        metadata_tags = _detect_instance_tags_from_metadata()
        if metadata_tags:
            return metadata_tags

    return []


def _resolve_runtime_credentials(require_tpu: bool, verbose: bool = False):
    config = EOConfig()
    project_id, zone, tpu_name = config.get_credentials()
    queued_resource = config.get_queued_resource()
    original = (project_id, zone, tpu_name, queued_resource)

    if not project_id:
        project_id = _detect_project_id_from_metadata()
        if project_id and verbose:
            console.print(f"[yellow]Auto-detected project ID from metadata: {project_id}[/yellow]")
    if not zone:
        zone = _detect_zone_from_metadata()
        if zone and verbose:
            console.print(f"[yellow]Auto-detected zone from metadata: {zone}[/yellow]")

    if not project_id:
        try:
            project_id = _read_gcloud_config_property("project")
            if verbose:
                console.print(f"[yellow]Using gcloud config project: {project_id}[/yellow]")
        except Exception:
            pass
    if not zone:
        try:
            zone = _read_gcloud_config_property("compute/zone")
            if verbose:
                console.print(f"[yellow]Using gcloud config zone: {zone}[/yellow]")
        except Exception:
            pass

    if project_id and zone:
        if not tpu_name:
            detected_tpu, detected_queued = _detect_tpu_identity_from_current_vm(project_id, zone)
            if detected_tpu:
                tpu_name = detected_tpu
                queued_resource = queued_resource or detected_queued
                if verbose:
                    console.print(f"[yellow]Auto-detected TPU name: {tpu_name}[/yellow]")
            elif queued_resource:
                resolved_tpu = _resolve_tpu_from_queued_resource(project_id, zone, queued_resource)
                if resolved_tpu:
                    tpu_name = resolved_tpu
                    if verbose:
                        console.print(f"[yellow]Resolved TPU name from queued resource: {tpu_name}[/yellow]")
        else:
            tpu_info = _describe_tpu_vm(project_id, zone, tpu_name)
            if tpu_info:
                current_queued = _basename(tpu_info.get("queuedResource"))
                if current_queued:
                    queued_resource = current_queued
            else:
                resolved_tpu = None
                if queued_resource:
                    resolved_tpu = _resolve_tpu_from_queued_resource(project_id, zone, queued_resource)
                if not resolved_tpu:
                    detected_tpu, detected_queued = _detect_tpu_identity_from_current_vm(project_id, zone)
                    resolved_tpu = detected_tpu
                    queued_resource = queued_resource or detected_queued
                if resolved_tpu:
                    if verbose:
                        console.print(
                            f"[yellow]Configured TPU '{tpu_name}' is stale; using '{resolved_tpu}' instead.[/yellow]"
                        )
                    tpu_name = resolved_tpu

    has_required = all([project_id, zone, tpu_name]) if require_tpu else all([project_id, zone])
    if not has_required:
        return None

    if (project_id, zone, tpu_name, queued_resource) != original and project_id and zone and tpu_name:
        config.set_credentials(project_id, zone, tpu_name, queued_resource=queued_resource)
        config.save_config()

    return config, project_id, zone, tpu_name, queued_resource


def _get_config_and_manager():
    resolved = _resolve_runtime_credentials(require_tpu=True, verbose=True)
    if not resolved:
        raise click.ClickException(
            "Could not resolve project/zone/tpu automatically. Run 'eopod configure' or pass values explicitly."
        )
    config, project_id, zone, tpu_name, _queued_resource = resolved
    tpu_manager = TPUManager(project_id, zone, tpu_name)
    return config, tpu_manager


@cli.command()
@click.option("--project-id", help="Google Cloud Project ID (optional if running on GCP)")
@click.option("--zone", help="Google Cloud Zone (optional if running on GCP)")
@click.option("--tpu-name", required=False, help="TPU Name (auto-detected on TPU VM if omitted)")
def configure(project_id, zone, tpu_name):
    """Configure eopod with your Google Cloud details"""
    config = EOConfig()
    stored_project_id, stored_zone, stored_tpu_name = config.get_credentials()
    queued_resource = config.get_queued_resource()

    if not project_id:
        project_id = _detect_project_id_from_metadata()
        if project_id:
            console.print(f"[yellow]Auto-detected project ID: {project_id}[/yellow]")
    if not project_id:
        try:
            project_id = _read_gcloud_config_property("project")
            console.print(f"[yellow]Using gcloud config project: {project_id}[/yellow]")
        except Exception:
            pass

    if not project_id and stored_project_id:
        project_id = stored_project_id
        console.print(f"[yellow]Using saved project ID: {project_id}[/yellow]")

    if not project_id:
        console.print("[red]Failed to auto-detect project ID. Please provide it manually.[/red]")
        return

    if not zone:
        zone = _detect_zone_from_metadata()
        if zone:
            console.print(f"[yellow]Auto-detected zone: {zone}[/yellow]")
    if not zone:
        try:
            zone = _read_gcloud_config_property("compute/zone")
            console.print(f"[yellow]Using gcloud config zone: {zone}[/yellow]")
        except Exception:
            pass

    if not zone and stored_zone:
        zone = stored_zone
        console.print(f"[yellow]Using saved zone: {zone}[/yellow]")

    if not zone:
        console.print("[red]Failed to auto-detect zone. Please provide it manually.[/red]")
        return

    if not tpu_name:
        detected_tpu, detected_queued = _detect_tpu_identity_from_current_vm(project_id, zone)
        if detected_tpu:
            tpu_name = detected_tpu
            queued_resource = detected_queued or queued_resource
            console.print(f"[yellow]Auto-detected TPU name: {tpu_name}[/yellow]")
        elif queued_resource:
            resolved_tpu = _resolve_tpu_from_queued_resource(project_id, zone, queued_resource)
            if resolved_tpu:
                tpu_name = resolved_tpu
                console.print(f"[yellow]Resolved TPU name from queued resource: {tpu_name}[/yellow]")
        elif stored_tpu_name:
            tpu_name = stored_tpu_name
            console.print(f"[yellow]Using saved TPU name: {tpu_name}[/yellow]")

    if not tpu_name:
        console.print("[red]Failed to auto-detect TPU name. Please provide --tpu-name manually.[/red]")
        return

    tpu_info = _describe_tpu_vm(project_id, zone, tpu_name)
    if not tpu_info and queued_resource:
        resolved_tpu = _resolve_tpu_from_queued_resource(project_id, zone, queued_resource)
        if resolved_tpu and resolved_tpu != tpu_name:
            console.print(f"[yellow]Saved TPU name '{tpu_name}' is stale; using '{resolved_tpu}'.[/yellow]")
            tpu_name = resolved_tpu
            tpu_info = _describe_tpu_vm(project_id, zone, tpu_name)

    if tpu_info:
        current_queued = _basename(tpu_info.get("queuedResource"))
        if current_queued:
            queued_resource = current_queued
    else:
        console.print(f"[yellow]Could not verify TPU '{tpu_name}' in {zone}. Saving configuration as provided.[/yellow]")

    config.set_credentials(project_id, zone, tpu_name, queued_resource=queued_resource)
    config.save_config()
    console.print("[green]Configuration saved successfully![/green]")
    if queued_resource:
        console.print(f"[green]Queued resource: {queued_resource}[/green]")


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
    _config, tpu_manager = _get_config_and_manager()
    try:
        internal_ips = await tpu_manager.get_internal_ips()
        tpu_manager.display_ips(internal_ips, "internal", output_format="comma")
    except Exception as e:
        console.print(f"[red]Failed to get internal IPs: {e!s}[/red]")


@cli.command()
@async_command
async def get_external_ips():
    """Get external IP addresses of TPU workers."""

    _config, tpu_manager = _get_config_and_manager()
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

    _config, tpu = _get_config_and_manager()

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
    _config, tpu = _get_config_and_manager()

    signal = "-9" if force else "-15"
    command = f"kill {signal} {pids}"

    returncode, _stdout, stderr = await tpu.execute_command(command, worker)

    if returncode == 0:
        console.print(f"[green]Successfully {'force ' if force else ''}killed process(es) {pids}[/green]")
    else:
        console.print(f"[red]Error killing process(es):[/red] {stderr}")


@cli.command()
@async_command
async def status():
    """Show TPU status and information"""
    _config, tpu = _get_config_and_manager()
    try:
        status = await tpu.get_status()
        network = _network_name(status.get("networkConfig", {}).get("network") or status.get("network"))

        table = Table(title="TPU Status")
        table.add_column("Property")
        table.add_column("Value")

        table.add_row("Name", status.get("name", ""))
        table.add_row("State", status.get("state", ""))
        table.add_row("Type", status.get("acceleratorType", ""))
        table.add_row("Network", network)
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
                returncode, stdout, _stderr = await tpu.execute_command(
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
                    returncode, _stdout, stderr = await tpu.execute_command(kill_cmd, worker=str(w), stream=False)
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
                    returncode, _stdout, stderr = await tpu.execute_command(cmd, worker=str(w), stream=False)
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
        network = _network_name(status.get("networkConfig", {}).get("network") or status.get("network"))

        table = Table(title="TPU Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", status.get("name", ""))
        table.add_row("State", status.get("state", ""))
        table.add_row("Type", status.get("acceleratorType", ""))
        table.add_row("Network", network)

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
    resolved = _resolve_runtime_credentials(require_tpu=False, verbose=False)
    if resolved:
        _config, project_id, zone, tpu_name, queued_resource = resolved
    else:
        config = EOConfig()
        project_id, zone, tpu_name = config.get_credentials()
        queued_resource = config.get_queued_resource()

    if project_id and zone:
        table = Table(title="Current Configuration")
        table.add_column("Setting")
        table.add_column("Value")

        table.add_row("Project ID", project_id)
        table.add_row("Zone", zone)
        table.add_row("TPU Name", tpu_name or "(unset)")
        table.add_row("Queued Resource", queued_resource or "(unset)")

        console.print(table)
    else:
        console.print("[red]No configuration found. Please run 'eopod configure' first.[/red]")


@cli.command()
def doctor():
    """Run environment diagnostics for eopod and TPU access."""
    checks = []

    def add_check(name: str, status: str, details: str, fix: str = ""):
        checks.append({"name": name, "status": status, "details": details, "fix": fix})

    gcloud_path = shutil.which("gcloud")
    if gcloud_path:
        add_check("gcloud", "PASS", f"Found at {gcloud_path}")
    else:
        add_check("gcloud", "FAIL", "gcloud CLI not found on PATH", "Install Google Cloud CLI and re-run doctor")

    active_account = None
    if gcloud_path:
        auth_result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
        )
        if auth_result.returncode == 0 and auth_result.stdout.strip():
            active_account = auth_result.stdout.strip().splitlines()[0]
            add_check("Auth", "PASS", f"Active account: {active_account}")
        else:
            add_check("Auth", "FAIL", "No active gcloud account", "Run: gcloud auth login")

    resolved = _resolve_runtime_credentials(require_tpu=False, verbose=False)
    if resolved:
        _config, project_id, zone, tpu_name, queued_resource = resolved
    else:
        cfg = EOConfig()
        project_id, zone, tpu_name = cfg.get_credentials()
        queued_resource = cfg.get_queued_resource()

    if project_id:
        add_check("Project", "PASS", f"Project ID: {project_id}")
    else:
        add_check(
            "Project",
            "FAIL",
            "Project ID is unset",
            "Run: eopod configure --project-id <PROJECT_ID> (or set gcloud project)",
        )

    if zone:
        add_check("Zone", "PASS", f"Zone: {zone}")
    else:
        add_check("Zone", "FAIL", "Zone is unset", "Run: eopod configure --zone <ZONE> (or set gcloud compute/zone)")

    if project_id and gcloud_path:
        api_result = subprocess.run(
            [
                "gcloud",
                "services",
                "list",
                "--enabled",
                f"--project={project_id}",
                "--filter=config.name:tpu.googleapis.com",
                "--format=value(config.name)",
            ],
            capture_output=True,
            text=True,
        )
        if api_result.returncode == 0 and "tpu.googleapis.com" in api_result.stdout:
            add_check("TPU API", "PASS", "tpu.googleapis.com is enabled")
        else:
            add_check(
                "TPU API",
                "FAIL",
                "tpu.googleapis.com not enabled (or not accessible)",
                f"Run: gcloud services enable tpu.googleapis.com --project={project_id}",
            )

    if tpu_name:
        detail = f"TPU name: {tpu_name}"
        if queued_resource:
            detail += f" (queued: {queued_resource})"
        add_check("TPU Name", "PASS", detail)
    else:
        add_check(
            "TPU Name",
            "FAIL",
            "TPU name is unset",
            "Run: eopod configure --tpu-name <TPU_NAME> (or run from within TPU VM for auto-detect)",
        )

    tpu_reachable = False
    if project_id and zone and tpu_name and gcloud_path:
        tpu_info = _describe_tpu_vm(project_id, zone, tpu_name)
        if tpu_info:
            state = tpu_info.get("state", "UNKNOWN")
            add_check("TPU Describe", "PASS", f"Reachable via API (state: {state})")
            tpu_reachable = True
        else:
            add_check(
                "TPU Describe",
                "FAIL",
                "Failed to describe TPU VM (missing permissions, wrong zone/name, or stale config)",
                "Re-run: eopod configure (will auto-refresh stale TPU names when possible)",
            )

    if tpu_reachable and project_id and zone and tpu_name:
        ssh_result = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                tpu_name,
                f"--project={project_id}",
                f"--zone={zone}",
                "--worker=0",
                "--command=true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if ssh_result.returncode == 0:
            add_check("TPU SSH", "PASS", "Non-interactive SSH command succeeded")
        else:
            ssh_error = (
                (ssh_result.stderr or ssh_result.stdout).strip().splitlines()[-1]
                if (ssh_result.stderr or ssh_result.stdout)
                else "SSH command failed"
            )
            add_check(
                "TPU SSH",
                "WARN",
                ssh_error,
                "Check VM networking/firewall, IAM, and SSH key setup. Re-run: gcloud compute tpus tpu-vm ssh ...",
            )

    table = Table(title="eopod Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")
    table.add_column("Suggested Fix")

    status_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    for item in checks:
        color = status_color.get(item["status"], "white")
        table.add_row(
            item["name"],
            f"[{color}]{item['status']}[/{color}]",
            item["details"],
            item["fix"],
        )

    pass_count = sum(1 for item in checks if item["status"] == "PASS")
    warn_count = sum(1 for item in checks if item["status"] == "WARN")
    fail_count = sum(1 for item in checks if item["status"] == "FAIL")

    console.print(table)
    console.print(
        f"[cyan]Summary:[/cyan] [green]{pass_count} pass[/green], "
        f"[yellow]{warn_count} warn[/yellow], [red]{fail_count} fail[/red]"
    )


@cli.command()
@click.option("--worker", default="all", help='Specific worker or "all"')
@click.option("--shell", default="/bin/bash", help="Shell to use (default: /bin/bash)")
def terminal(worker, shell):
    """Open an interactive terminal session with TPU workers"""
    _ = shell
    resolved = _resolve_runtime_credentials(require_tpu=True, verbose=True)
    if not resolved:
        console.print("[red]Could not resolve project/zone/tpu automatically. Run 'eopod configure' first.[/red]")
        return
    _config, project_id, zone, tpu_name, _queued_resource = resolved

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
    _config, tpu = _get_config_and_manager()

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
    _config, tpu = _get_config_and_manager()
    command = r"""
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
                spot_tpu_project_id = _read_gcloud_config_property("project")
                console.print(f"[green]Using current project for spot TPU: {spot_tpu_project_id}[/green]")

            if not spot_tpu_zone:
                spot_tpu_zone = _read_gcloud_config_property("compute/zone")
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
                "eformer.executor.patch_tpus_ray",
                "--self-job",
                "--head-only" if not head_has_tpu else None,
                "--head-node-ip",
                head_ip,
            ]
            cmd = [part for part in cmd if part]

            subprocess.run(" ".join(cmd), shell=True, check=True)
            console.print("[green]Ray head started successfully[/green]")
            time.sleep(5)

        console.print(f"[cyan]Configuring {len(spot_internal_ips)} spot TPU workers...[/cyan]")

        cmd_parts = [
            python_path or PYTHON_PATH,
            "-m",
            "eformer.executor.patch_tpus_ray",
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
                "eformer.executor.patch_tpus_ray",
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
            "eformer.executor.patch_tpus_ray",
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
    help="Network tag for VMs. If omitted, eopod auto-detects the TPU VM tag from GCP.",
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
    resolved = _resolve_runtime_credentials(require_tpu=True, verbose=True)
    if not resolved:
        console.print("[red]Could not resolve project/zone/tpu automatically. Run 'eopod configure' first.[/red]")
        return
    _config, project_id, zone, tpu_name, _queued_resource = resolved

    safe_tpu_name = tpu_name.lower().replace("_", "-")

    tpu_manager = TPUManager(project_id, zone, tpu_name)
    tpu_info = None

    try:
        tpu_info = await tpu_manager.get_tpu_info()
    except Exception as e:
        console.print(f"[yellow]Could not fetch TPU details automatically: {e}[/yellow]")

    if network is None:
        try:
            raw_network = (tpu_info or {}).get("networkConfig", {}).get("network", "default")
            network = _network_name(raw_network)
            console.print(f"Using network: {network}")
        except Exception as e:
            console.print(f"[yellow]Could not determine network from TPU config: {e}[/yellow]")
            console.print("[yellow]Using 'default' network instead[/yellow]")
            network = "default"

    detected_target_tags = []
    if target_tag is None:
        detected_target_tags = _resolve_tpu_vm_target_tags(project_id, zone, tpu_name, tpu_info=tpu_info)
        if detected_target_tags:
            console.print(f"[green]Auto-detected TPU VM target tag(s): {', '.join(detected_target_tags)}[/green]")
        else:
            detected_target_tags = [safe_tpu_name]
            console.print(
                "[yellow]Could not auto-detect the TPU VM target tag. "
                f"Falling back to '{safe_tpu_name}'. If the port still does not open, "
                "pass --target-tag explicitly.[/yellow]"
            )
    else:
        detected_target_tags = _normalize_target_tags(target_tag.split(","))
        console.print(f"[green]Using user-provided target tag(s): {', '.join(detected_target_tags)}[/green]")

    if verify_tag:
        try:
            vm_tags = _resolve_tpu_vm_target_tags(project_id, zone, tpu_name, tpu_info=tpu_info)
            missing_tags = [tag for tag in detected_target_tags if tag not in vm_tags]
            if missing_tags:
                console.print(f"[red]Target tag(s) '{', '.join(missing_tags)}' are not applied to the TPU VM![/red]")
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
                existing_rule_raw = await run_command(cmd, capture_output=True)
                existing_rule = json.loads(existing_rule_raw)
                rule_exists = True
                console.print(f"Rule '{rule_name}' already exists.")

                existing_target_tags = _normalize_target_tags(existing_rule.get("targetTags", []))
                if sorted(existing_target_tags) != sorted(detected_target_tags):
                    console.print(
                        "[yellow]Existing rule uses stale target tags. "
                        "Updating it to match the TPU VM automatically.[/yellow]"
                    )
                elif not update_existing:
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
                f"--priority={priority}",
            ]

            if not rule_exists:
                cmd_parts.extend(
                    [
                        f"--direction={current_direction.upper()}",
                        f"--network={network}",
                        "--action=ALLOW",
                    ]
                )

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

            cmd_parts.append(f"--target-tags={','.join(detected_target_tags)}")

            cmd_parts.append(f"--description={shlex.quote(description)}")

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
