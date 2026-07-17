from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_package.py"
SPEC = importlib.util.spec_from_file_location("verify_package", SCRIPT)
verify_package = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_package)


def _member(name, kind=tarfile.REGTYPE):
    member = tarfile.TarInfo(name)
    member.type = kind
    return member


def test_sdist_validator_accepts_one_normal_root():
    files = verify_package._validated_sdist_files([
        _member("videowipe-0.5.0", tarfile.DIRTYPE),
        _member("videowipe-0.5.0/LICENSE"),
    ])

    assert files == ["LICENSE"]


@pytest.mark.parametrize(
    "members",
    [
        [_member("videowipe-0.5.0/../escape")],
        [_member("videowipe-0.5.0/link", tarfile.SYMTYPE)],
        [_member("one/LICENSE"), _member("two/README.md")],
    ],
)
def test_sdist_validator_rejects_unsafe_members(members):
    with pytest.raises(SystemExit):
        verify_package._validated_sdist_files(members)
