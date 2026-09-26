from __future__ import annotations

import base64
import hashlib
import importlib.util
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

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


smoke_spec = importlib.util.spec_from_file_location(
    "smoke_installed", Path(__file__).parents[1] / "scripts/smoke_installed.py")
smoke = importlib.util.module_from_spec(smoke_spec)
smoke_spec.loader.exec_module(smoke)

@pytest.mark.parametrize("case", ["good", "contrib", "installed_contrib", "no_provider", "shadow", "mismatch", "no_hash", "no_record", "no_native", "cocoa"])
def test_headless_native_provenance(tmp_path, monkeypatch, case):
    native = tmp_path / "cv2.abi3.so"
    native.write_bytes(b"native binary")
    digest = base64.urlsafe_b64encode(hashlib.sha256(native.read_bytes()).digest()).rstrip(b"=").decode()
    record = SimpleNamespace(locate=lambda: native,
                             hash=None if case == "no_hash" else SimpleNamespace(mode="sha256", value=digest))
    providers = ["opencv-python-headless"]
    if case == "contrib":
        providers.append("opencv-contrib-python")
    installed = providers + (["opencv-contrib-python"] if case == "installed_contrib" else [])
    if case == "no_provider":
        providers = []
    metadata = smoke.importlib.metadata
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"cv2": providers})
    monkeypatch.setattr(metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name": name}) for name in installed])
    monkeypatch.setattr(metadata, "distribution", lambda name: SimpleNamespace(files=None if case == "no_record" else [record]))
    loaded = tmp_path / "shadow.so" if case == "shadow" else native
    fake = SimpleNamespace(_native=SimpleNamespace(__file__=str(loaded)), __file__=str(tmp_path / "__init__.py"),
                           getBuildInformation=lambda: "GUI: COCOA" if case == "cocoa" else "GUI: NONE")
    if case == "no_native":
        del fake._native
    if case == "mismatch":
        native.write_bytes(b"overwritten binary")
    monkeypatch.setattr(smoke, "cv2", fake)
    monkeypatch.setattr(sys, "platform", "darwin")
    check = smoke._check_headless
    if case == "good":
        check()
    else:
        with pytest.raises(SystemExit):
            check()
