import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from eopod import EOConfig, cli  


# Mock gcloud calls for testing
@pytest.fixture(autouse=True)
def mock_subprocess_exec():
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        # Default mock behavior (can be overridden in individual tests)
        mock_exec.return_value = AsyncMock(
            communicate=AsyncMock(return_value=(b"{}", b"")), returncode=0
        )  # Successful return
        yield mock_exec


@pytest.fixture
def config(tmp_path):
    """Creates a temporary config directory for testing."""
    config = EOConfig()
    config.config_dir = tmp_path / ".eopod"
    config.config_file = config.config_dir / "config.ini"
    config.history_file = config.config_dir / "history.yaml"
    config.error_log_file = config.config_dir / "error_log.yaml"
    config.ensure_config_dir()
    return config


def test_configure_command(config, runner):
    """Tests the 'configure' command."""
    result = runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    assert result.exit_code == 0
    assert "Configuration saved successfully!" in result.output

    config.config.read(config.config_file)
    assert config.config["DEFAULT"]["project_id"] == "test-project"
    assert config.config["DEFAULT"]["zone"] == "test-zone"
    assert config.config["DEFAULT"]["tpu-name"] == "test-tpu"


@pytest.mark.asyncio
async def test_status_command_success(config, runner, mock_subprocess_exec):
    """Tests the 'status' command with successful output."""
    mock_subprocess_exec.return_value = AsyncMock(
        communicate=AsyncMock(
            return_value=(
                b'{"name": "test-tpu", "state": "READY", "acceleratorType": "v3-8", "network": "test-network", "apiVersion": "V2"}',
                b"",
            )
        ),
        returncode=0,
    )

    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "test-tpu" in result.output
    assert "READY" in result.output


@pytest.mark.asyncio
async def test_status_command_failure(config, runner, mock_subprocess_exec):
    """Tests the 'status' command with failed output."""
    mock_subprocess_exec.return_value = AsyncMock(
        communicate=AsyncMock(return_value=(b"", b"TPU not found")), returncode=1
    )

    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Failed to get TPU status" in result.output


@pytest.mark.asyncio
async def test_run_command_success(config, runner, mock_subprocess_exec):
    """Tests the 'run' command with a successful execution."""
    mock_subprocess_exec.return_value = AsyncMock(
        communicate=AsyncMock(return_value=(b"Command output", b"")), returncode=0
    )

    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(cli, ["run", "echo hello"])
    assert result.exit_code == 0
    assert "Command completed successfully!" in result.output
    assert "Command output" in result.output

    with open(config.history_file, "r") as f:
        history = yaml.safe_load(f)
        assert history[-1]["command"] == "echo hello"
        assert history[-1]["status"] == "success"


@pytest.mark.asyncio
async def test_run_command_retry_and_failure(config, runner, mock_subprocess_exec):
    """Tests the 'run' command with retries and eventual failure."""
    mock_subprocess_exec.return_value = AsyncMock(
        communicate=AsyncMock(return_value=(b"", b"Command failed")), returncode=1
    )

    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(
        cli, ["run", "some_bad_command", "--retry", "2", "--delay", "1"]
    )
    assert result.exit_code == 0
    assert "Attempt 1 failed" in result.output
    assert "Attempt 2 failed" in result.output
    assert "Command failed after 2 attempts" in result.output

    with open(config.error_log_file, "r") as f:
        error_log = yaml.safe_load(f)
        assert len(error_log) >= 2
        assert error_log[-1]["command"] == "some_bad_command"


@pytest.mark.asyncio
async def test_run_command_timeout(config, runner, mock_subprocess_exec):
    """Tests the 'run' command with a timeout."""

    async def slow_communicate(*args, **kwargs):
        await asyncio.sleep(2)  # Simulate a long-running command
        return (b"", b"")

    mock_subprocess_exec.return_value = AsyncMock(
        communicate=AsyncMock(side_effect=slow_communicate), returncode=0
    )

    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(
        cli, ["run", "sleep 3", "--timeout", "1"]
    )  # Timeout after 1 second
    assert result.exit_code == 0
    assert "Command timed out after 1 seconds" in result.output

    with open(config.error_log_file, "r") as f:
        error_log = yaml.safe_load(f)
        assert error_log[-1]["command"] == "sleep 3"
        assert error_log[-1]["error"] == "Command timed out"


def test_history_command(config, runner):
    """Tests the 'history' command."""
    # Add some history entries
    config.save_command_history("command 1", "success", "output 1")
    config.save_command_history("command 2", "failed", "output 2")

    result = runner.invoke(cli, ["history"])
    assert result.exit_code == 0
    assert "command 1" in result.output
    assert "command 2" in result.output
    assert "success" in result.output
    assert "failed" in result.output


def test_errors_command(config, runner):
    """Tests the 'errors' command."""
    # Add some error log entries
    config.save_error_log("error command 1", "error 1 details")
    config.save_error_log("error command 2", "error 2 details")

    result = runner.invoke(cli, ["errors"])
    assert result.exit_code == 0
    assert "error command 1" in result.output
    assert "error command 2" in result.output
    assert "error 1 details" in result.output
    assert "error 2 details" in result.output


def test_show_config_command(config, runner):
    """Tests the 'show-config' command."""
    runner.invoke(
        cli,
        [
            "configure",
            "--project-id",
            "test-project",
            "--zone",
            "test-zone",
            "--tpu-name",
            "test-tpu",
        ],
    )
    result = runner.invoke(cli, ["show-config"])
    assert result.exit_code == 0
    assert "test-project" in result.output
    assert "test-zone" in result.output
    assert "test-tpu" in result.output


def test_no_config(runner):
    """Tests behavior when no configuration is found."""
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Please configure EOpod first" in result.output

    result = runner.invoke(cli, ["run", "ls"])
    assert result.exit_code == 0
    assert "Please configure EOpod first" in result.output
