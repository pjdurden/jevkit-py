"""The pytest plugin: fixtures and the ``--jev-record`` flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from .assertions import assert_answer
from .cassette import Cassette

__all__ = ["jev_cassette", "jev_cassette_dir", "jev_client"]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("jev", "TypeSafe Jev cassettes")
    group.addoption(
        "--jev-record", action="store_true", default=False,
        help="record any request missing from its cassette (mode 'auto')",
    )
    group.addoption(
        "--jev-rerecord", action="store_true", default=False,
        help="re-record every request, replacing the cassettes (mode 'record')",
    )
    group.addoption(
        "--jev-cassette-dir", action="store", default=None,
        help="where cassettes live (default: tests/cassettes next to the test file)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "jev_cassette(name): use a named cassette file")


def _mode(config: pytest.Config) -> str:
    if config.getoption("--jev-rerecord"):
        return "record"
    if config.getoption("--jev-record"):
        return "auto"
    return "replay"


@pytest.fixture(scope="session")
def jev_cassette_dir(request: pytest.FixtureRequest) -> Path:
    """Directory holding cassettes. Override in a conftest to relocate them."""
    configured = request.config.getoption("--jev-cassette-dir")
    if configured:
        return Path(configured)
    return Path(str(request.config.rootpath)) / "tests" / "cassettes"


@pytest.fixture
def jev_client() -> Any:
    """The live client, used only when recording.

    Override this in your conftest to return something with a ``system_one``
    method. Left as ``None`` a replay-only run still works; a recording run
    fails with a message saying exactly this.
    """
    return None


@pytest.fixture
def jev_cassette(
    request: pytest.FixtureRequest,
    jev_cassette_dir: Path,
    jev_client: Any,
) -> Iterator[Cassette]:
    """A cassette scoped to the current test.

    The file is named after the test by default, so one test's recordings never
    collide with another's. Override with ``@pytest.mark.jev_cassette("name")``
    to share one cassette across several tests.
    """
    marker = request.node.get_closest_marker("jev_cassette")
    name = marker.args[0] if marker and marker.args else request.node.name
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

    jev_cassette_dir.mkdir(parents=True, exist_ok=True)
    path = jev_cassette_dir / f"{safe}.jevl"

    system_one = getattr(jev_client, "system_one", None) if jev_client is not None else None
    cassette = Cassette(path, system_one, mode=_mode(request.config))
    yield cassette


# Re-exported so ``from jevkit_pytest import assert_answer`` works after the
# plugin is loaded as an entry point.
__all__ += ["assert_answer"]
