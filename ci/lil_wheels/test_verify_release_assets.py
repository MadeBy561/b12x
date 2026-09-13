"""Tests for the immutable B12X release-asset verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_COMMIT = "a" * 40
BETA_TAG = f"b12x-cu134-beta-{SOURCE_COMMIT}"
WHEEL_NAME = "b12x-1.3.0-py3-none-any.whl"


def digest(data: bytes) -> str:
    """Return the SHA-256 digest for generated fixture bytes."""
    return hashlib.sha256(data).hexdigest()


class VerifyReleaseAssetsTest(unittest.TestCase):
    """Exercise complete beta and promotion asset contracts."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.script = Path(__file__).with_name("verify_release_assets.py")
        self._write_assets(promotion=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_assets(self, *, promotion: bool) -> None:
        wheel = b"source-locked B12X wheel"
        files = {
            WHEEL_NAME: wheel,
            "install.sh": b"#!/bin/sh\n",
            "requirements-github.txt": b"b12x @ https://example.invalid/wheel\n",
            "runtime.lock": b"cxx11-abi=1\n",
        }
        for name, data in files.items():
            (self.directory / name).write_bytes(data)
        manifest = {
            "schema": "local-inference-b12x-wheel-release/v1",
            "source": {"commit": SOURCE_COMMIT},
            "release_tag": BETA_TAG,
            "packages": [
                {
                    "name": "b12x",
                    "file": WHEEL_NAME,
                    "sha256": digest(wheel),
                }
            ],
        }
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        checksummed = {
            "manifest.json": manifest_path.read_bytes(),
            "requirements-github.txt": files["requirements-github.txt"],
            "runtime.lock": files["runtime.lock"],
            "install.sh": files["install.sh"],
            f"wheels/{WHEEL_NAME}": wheel,
        }
        (self.directory / "SHA256SUMS").write_text(
            "".join(f"{digest(data)}  {name}\n" for name, data in checksummed.items())
        )
        archive_name = f"b12x-cu134-{SOURCE_COMMIT}.tar.zst"
        archive = b"deterministic release archive"
        (self.directory / archive_name).write_bytes(archive)
        (self.directory / f"{archive_name}.sha256").write_text(
            f"{digest(archive)}  {archive_name}\n"
        )
        if promotion:
            promotion_marker = {
                "schema": "local-inference-b12x-promotion/v1",
                "status": "byte-identical-promotion",
                "source_release": BETA_TAG,
                "source_commit": SOURCE_COMMIT,
                "source_manifest_sha256": digest(manifest_path.read_bytes()),
            }
            (self.directory / "stable-promotion.json").write_text(
                json.dumps(promotion_marker)
            )

    def _verify(self, *, promotion: bool = False) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(self.script),
            "--directory",
            str(self.directory),
            "--source-commit",
            SOURCE_COMMIT,
            "--beta-tag",
            BETA_TAG,
        ]
        if promotion:
            command.append("--promotion")
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_complete_beta_release_passes(self) -> None:
        """A complete beta release satisfies every declared digest."""
        result = self._verify()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unexpected_asset_is_rejected(self) -> None:
        """An undeclared file cannot be included in a release."""
        (self.directory / "unexpected.bin").write_bytes(b"data")
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release asset set differs", result.stderr)

    def test_modified_wheel_is_rejected(self) -> None:
        """Wheel bytes must match the manifest and checksum inventory."""
        (self.directory / WHEEL_NAME).write_bytes(b"modified")
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256SUMS mismatch", result.stderr)

    def test_complete_promotion_passes(self) -> None:
        """A promotion marker binds stable assets to one beta manifest."""
        self._write_assets(promotion=True)
        result = self._verify(promotion=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
