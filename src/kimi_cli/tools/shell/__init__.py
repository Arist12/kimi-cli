import asyncio
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import override

import kaos
from kaos import AsyncReadable
from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import ShellDisplayBlock
from kimi_cli.tools.utils import ToolRejectedError, ToolResultBuilder, load_desc
from kimi_cli.utils.environment import Environment
from kimi_cli.utils.subprocess_env import get_clean_env

MAX_TIMEOUT = int(os.environ.get("KIMI_MAX_TIMEOUT", 90 * 60))


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _shell_env_provider() -> str:
    provider = os.environ.get("KIMI_ENV_PROVIDER", "").strip().lower()
    if provider:
        return provider
    # AMDPilot passes this knob during dry-run-amdspace experiments. Treat it as
    # an explicit provider request so a missing amdspace install fails closed
    # instead of silently running the real Docker shell.
    if _truthy_env("AMDPILOT_AMDSPACE"):
        return "amdspace"
    return "docker"


def _float_env(*names: str, default: float) -> float:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                return default
    return default


def _int_env(*names: str, default: int | None = None) -> int | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                return default
    return default


def _add_amdspace_pythonpath() -> None:
    raw = (
        os.environ.get("KIMI_AMDSPACE_PYTHONPATH", "").strip()
        or os.environ.get("AMDPILOT_AMDSPACE_PYTHONPATH", "").strip()
    )
    for entry in reversed([p for p in raw.split(os.pathsep) if p]):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _amdspace_seed(command: str) -> int:
    explicit = _int_env("KIMI_AMDSPACE_SEED", "AMDPILOT_AMDSPACE_SEED", default=None)
    if explicit is not None:
        return explicit
    task_id = os.environ.get("AMDPILOT_TASK_ID", "") or os.environ.get("AMDPILOT_EXPERIMENT_ID", "")
    digest = hashlib.sha256(f"{task_id}\0{command}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


class Params(BaseModel):
    command: str = Field(description="The bash command to execute.")
    timeout: int = Field(
        description=(
            "The timeout in seconds for the command to execute. "
            "If the command takes longer than this, it will be killed."
        ),
        default=60,
        ge=1,
        le=MAX_TIMEOUT,
    )


class Shell(CallableTool2[Params]):
    name: str = "Shell"
    params: type[Params] = Params

    def __init__(self, approval: Approval, environment: Environment):
        is_powershell = environment.shell_name == "Windows PowerShell"
        super().__init__(
            description=load_desc(
                Path(__file__).parent / ("powershell.md" if is_powershell else "bash.md"),
                {"SHELL": f"{environment.shell_name} (`{environment.shell_path}`)"},
            )
        )
        self._approval = approval
        self._is_powershell = is_powershell
        self._shell_path = environment.shell_path

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder()

        if not params.command:
            return builder.error("Command cannot be empty.", brief="Empty command")

        if not await self._approval.request(
            self.name,
            "run command",
            f"Run command `{params.command}`",
            display=[
                ShellDisplayBlock(
                    language="powershell" if self._is_powershell else "bash",
                    command=params.command,
                )
            ],
        ):
            return ToolRejectedError()

        def stdout_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)

        def stderr_cb(line: bytes):
            line_str = line.decode(encoding="utf-8", errors="replace")
            builder.write(line_str)

        try:
            exitcode = await self._run_shell_command(
                params.command, stdout_cb, stderr_cb, params.timeout
            )

            if exitcode == 0:
                return builder.ok("Command executed successfully.")
            else:
                return builder.error(
                    f"Command failed with exit code: {exitcode}.",
                    brief=f"Failed with exit code: {exitcode}",
                )
        except TimeoutError:
            return builder.error(
                f"Command killed by timeout ({params.timeout}s)",
                brief=f"Killed by timeout ({params.timeout}s)",
            )

    async def _run_shell_command(
        self,
        command: str,
        stdout_cb: Callable[[bytes], None],
        stderr_cb: Callable[[bytes], None],
        timeout: int,
    ) -> int:
        provider = _shell_env_provider()
        if provider == "amdspace":
            return await self._run_amdspace_command(command, stdout_cb, stderr_cb, timeout)
        if provider not in ("", "docker", "real"):
            stderr_cb(f"KIMI_ENV_PROVIDER={provider!r} is not supported\n".encode())
            return 127

        async def _read_stream(stream: AsyncReadable, cb: Callable[[bytes], None]):
            while True:
                line = await stream.readline()
                if line:
                    cb(line)
                else:
                    break

        process = await kaos.exec(*self._shell_args(command), env=get_clean_env())

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(process.stdout, stdout_cb),
                    _read_stream(process.stderr, stderr_cb),
                ),
                timeout,
            )
            return await process.wait()
        except TimeoutError:
            await process.kill()
            raise

    async def _run_amdspace_command(
        self,
        command: str,
        stdout_cb: Callable[[bytes], None],
        stderr_cb: Callable[[bytes], None],
        timeout: int,
    ) -> int:
        try:
            _add_amdspace_pythonpath()
            from amdspace import TeacherConfig, TeacherEnv, ToolCall  # type: ignore
        except Exception as exc:  # noqa: BLE001 - explicit simulator mode fails closed
            stderr_cb(f"amdspace environment provider unavailable: {exc}\n".encode())
            return 127

        db_path = (
            os.environ.get("KIMI_AMDSPACE_DB", "").strip()
            or os.environ.get("AMDPILOT_AMDSPACE_DB", "").strip()
            or None
        )
        try:
            response = TeacherEnv(
                TeacherConfig(
                    seed=_amdspace_seed(command),
                    db_path=db_path,
                    real_case_rate=_float_env(
                        "KIMI_AMDSPACE_REAL_CASE_RATE",
                        "AMDPILOT_AMDSPACE_REAL_CASE_RATE",
                        default=0.35,
                    ),
                    failure_rate=_float_env(
                        "KIMI_AMDSPACE_FAILURE_RATE",
                        "AMDPILOT_AMDSPACE_FAILURE_RATE",
                        default=0.18,
                    ),
                    noise_rate=_float_env(
                        "KIMI_AMDSPACE_NOISE_RATE",
                        "AMDPILOT_AMDSPACE_NOISE_RATE",
                        default=0.12,
                    ),
                )
            ).respond(ToolCall("Shell", {"command": command, "timeout": timeout}))
        except Exception as exc:  # noqa: BLE001
            stderr_cb(f"amdspace environment provider failed: {exc}\n".encode())
            return 127

        # Preserve the Shell tool contract: stdout/stderr are merged into the
        # tool output, and the returned exit code drives ToolResultBuilder.ok/error.
        if (
            getattr(response, "latency_s", 0) > 0
            and (_truthy_env("KIMI_AMDSPACE_APPLY_LATENCY") or _truthy_env("AMDPILOT_AMDSPACE_APPLY_LATENCY"))
        ):
            await asyncio.sleep(min(float(response.latency_s), max(0, timeout)))
        stdout_cb((response.as_tool_output().rstrip() + "\n").encode("utf-8", errors="replace"))
        return int(getattr(response, "exit_code", 0) or 0)

    def _shell_args(self, command: str) -> tuple[str, ...]:
        if self._is_powershell:
            return (str(self._shell_path), "-command", command)
        return (str(self._shell_path), "-c", command)
