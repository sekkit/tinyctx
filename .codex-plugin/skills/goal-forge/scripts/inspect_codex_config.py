#!/usr/bin/env python3
"""Read-only Codex config readiness report for long-running /goal work."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


CONFIG_PATH = Path.home() / ".codex" / "config.toml"
# Codex CLI v0.128.0 introduced persisted /goal workflows.
MIN_GOAL_VERSION = (0, 128, 0)
MIN_GOAL_VERSION_LABEL = ".".join(str(part) for part in MIN_GOAL_VERSION)
AUTONOMOUS_SETTINGS = {
    "model": "gpt-5.5",
    "model_context_window": 1050000,
    "model_auto_compact_token_limit": 997500,
    "model_reasoning_effort": "high",
    "plan_mode_reasoning_effort": "xhigh",
    "approval_policy": "never",
    "sandbox_mode": "danger-full-access",
}
TINYCTX_GOAL_SETTINGS = {
    "model_provider": "tinyctx",
    "model": "tinyctx-auto",
    "model_context_window": 1050000,
    "model_auto_compact_token_limit": 997500,
    "model_reasoning_effort": "high",
    "plan_mode_reasoning_effort": "xhigh",
    "approval_policy": "never",
    "sandbox_mode": "danger-full-access",
}


def codex_command_candidates(config: dict) -> list[str]:
    candidates = []
    env_path = os.environ.get("CODEX_CLI_PATH")
    if env_path:
        candidates.append(env_path)

    mcp_servers = config.get("mcp_servers")
    if isinstance(mcp_servers, dict):
        node_repl = mcp_servers.get("node_repl")
        if isinstance(node_repl, dict):
            node_env = node_repl.get("env")
            if isinstance(node_env, dict):
                configured_path = node_env.get("CODEX_CLI_PATH")
                if isinstance(configured_path, str) and configured_path:
                    candidates.append(configured_path)

    candidates.append("codex")
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def codex_version(config: dict) -> tuple[str | None, str | None]:
    for command in codex_command_candidates(config):
        try:
            result = subprocess.run(
                [command, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), command
    return None, None


def codex_feature_state(config: dict, feature: str,
                        profile_name: str | None) -> str | None:
    """Ask the installed CLI for the effective feature state when possible."""
    version_command = codex_version(config)[1]
    if not version_command:
        return None
    cmd = [version_command]
    if profile_name:
        cmd.extend(["--profile", profile_name])
    cmd.extend(["features", "list"])
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == feature:
            return parts[-1]
    return None


def parse_version(raw: str | None) -> tuple[int, int, int] | None:
    if not raw:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def load_config() -> tuple[dict, str | None]:
    if not CONFIG_PATH.exists():
        return {}, "missing"
    try:
        with CONFIG_PATH.open("rb") as handle:
            return tomllib.load(handle), None
    except tomllib.TOMLDecodeError as exc:
        return {}, f"invalid TOML: {exc}"
    except OSError as exc:
        return {}, f"could not read config: {exc}"


def current_project_trust(config: dict, project_path: Path) -> str | None:
    target_path = project_path.resolve()
    projects = config.get("projects")
    if not isinstance(projects, dict):
        return None

    best_match: tuple[int, str] | None = None
    for raw_path, settings in projects.items():
        try:
            registered_path = Path(os.path.expanduser(raw_path)).resolve()
        except OSError:
            continue
        if not isinstance(settings, dict):
            continue
        try:
            target_path.relative_to(registered_path)
        except ValueError:
            continue
        trust_level = settings.get("trust_level")
        if isinstance(trust_level, str):
            score = len(registered_path.parts)
            if best_match is None or score > best_match[0]:
                best_match = (score, trust_level)
    return best_match[1] if best_match else None


def profile_overrides(config: dict) -> list[tuple[str, str, object]]:
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return []

    overrides: list[tuple[str, str, object]] = []
    for profile_name, settings in sorted(profiles.items()):
        if not isinstance(settings, dict):
            continue
        for key in (
            "model_provider",
            "approval_policy",
            "sandbox_mode",
            "model",
            "model_context_window",
            "model_auto_compact_token_limit",
            "model_reasoning_effort",
            "plan_mode_reasoning_effort",
        ):
            if key in settings:
                overrides.append((str(profile_name), key, settings[key]))
        features = settings.get("features")
        if isinstance(features, dict) and "goals" in features:
            overrides.append((str(profile_name), "features.goals",
                              features["goals"]))
    return overrides


def print_key(config: dict, key: str) -> None:
    value = config.get(key, "<unset>")
    print(f"{key}: {value}")


def config_value(config: dict, key: str) -> object:
    return config.get(key, "<unset>")


def merge_config_with_profile(config: dict, profile_name: str | None) -> dict:
    if not profile_name:
        return dict(config)
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return dict(config)
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return dict(config)

    def merge(base: dict, overlay: dict) -> dict:
        merged = dict(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return merge(config, profile)


def expected_settings(profile_name: str | None) -> dict:
    if profile_name == "tinyctx-goal":
        return TINYCTX_GOAL_SETTINGS
    return AUTONOMOUS_SETTINGS


def autonomous_gaps(config: dict, features: dict, trust_level: str | None,
                    expected: dict) -> list[str]:
    gaps = []
    for key, expected_value in expected.items():
        actual = config_value(config, key)
        if actual != expected_value:
            gaps.append(f"{key}: expected {expected_value!r}, got {actual!r}")
    if features.get("goals") is not True:
        gaps.append(f"features.goals: expected True, got {features.get('goals', '<unset>')!r}")
    if trust_level != "trusted":
        gaps.append(f"current_project_trust: expected 'trusted', got {trust_level or '<unset>'!r}")
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Codex config readiness report for long-running /goal work.",
    )
    parser.add_argument(
        "--project-path",
        default=".",
        help="Project path to check against Codex trust settings. Defaults to cwd.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Codex profile to evaluate after profile overrides, e.g. tinyctx-goal.",
    )
    args = parser.parse_args()
    project_path = Path(args.project_path).expanduser()

    config, config_error = load_config()
    version_raw, version_command = codex_version(config)
    version = parse_version(version_raw)
    selected_profile = args.profile
    if selected_profile is None and isinstance(config.get("profile"), str):
        selected_profile = str(config["profile"])
    effective_config = merge_config_with_profile(config, selected_profile)
    raw_features = effective_config.get("features")
    features = raw_features if isinstance(raw_features, dict) else {}
    profiles = config.get("profiles")
    profile_missing = (
        selected_profile is not None and (
            not isinstance(profiles, dict) or selected_profile not in profiles
        )
    )

    print("Codex /goal readiness report")
    print()
    print(f"codex --version: {version_raw or '<not found>'}")
    print(f"codex_command: {version_command or '<not found>'}")
    if version and version < MIN_GOAL_VERSION:
        print(f"version_status: update recommended; /goal was introduced in codex-cli {MIN_GOAL_VERSION_LABEL}")
    elif version:
        print("version_status: likely new enough for /goal")
    else:
        print("version_status: could not determine")

    print()
    print(f"config_path: {CONFIG_PATH}")
    print(f"config_exists: {CONFIG_PATH.exists()}")
    print(f"config_status: {config_error or 'ok'}")
    print(f"selected_profile: {selected_profile or '<top-level>'}")
    if profile_missing:
        print("profile_status: missing")
    trust_level = current_project_trust(config, project_path)
    print(f"project_path: {project_path.resolve()}")
    print(f"current_project_trust: {trust_level or '<unset>'}")
    print_key(effective_config, "model_provider")
    print_key(effective_config, "model")
    print_key(effective_config, "model_context_window")
    print_key(effective_config, "model_reasoning_effort")
    print_key(effective_config, "plan_mode_reasoning_effort")
    print_key(effective_config, "model_auto_compact_token_limit")
    print_key(effective_config, "approval_policy")
    print_key(effective_config, "sandbox_mode")
    print(f"features.goals: {features.get('goals', '<unset>')}")
    cli_goal_state = codex_feature_state(config, "goals", selected_profile)
    if cli_goal_state is not None:
        print(f"codex_features_list.goals: {cli_goal_state}")
    overrides = profile_overrides(config)
    if overrides:
        print("profile_overrides:")
        for profile_name, key, value in overrides:
            print(f"- profiles.{profile_name}.{key}: {value}")
    else:
        print("profile_overrides: <none>")

    print()
    gaps = autonomous_gaps(
        effective_config, features, trust_level,
        expected_settings(selected_profile))
    if profile_missing:
        gaps.insert(0, f"profile: expected {selected_profile!r} to exist")
    if gaps:
        print("autonomous_goal_status: not ready")
        print("autonomous_goal_gaps:")
        for gap in gaps:
            print(f"- {gap}")
    else:
        print("autonomous_goal_status: ready")

    print()
    print("notes:")
    print("- This script is read-only.")
    print("- Use model_reasoning_effort=high for execution and plan_mode_reasoning_effort=xhigh for planning.")
    print("- model_auto_compact_token_limit=997500 lets long /goal sessions compact before the context limit.")
    print("- Use approval_policy=never and sandbox_mode=danger-full-access only in explicitly trusted project directories.")
    print("- Do not start a long goal until SPEC.md has user-approved measurable done_when criteria, a scorecard, a fast feedback loop, and working-memory files when needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
