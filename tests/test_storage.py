"""Tests for ``dark_factory.storage.backend`` — LocalStorage, S3Storage, RunStorage."""

from __future__ import annotations



import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dark_factory.storage.backend import (
    LocalStorage,
    RunStorage,
    S3Storage,
    StorageBackend,
    get_storage,
    reset_storage,
)


# ── LocalStorage ──────────────────────────────────────────────────────────────


class TestLocalStorage:
    def test_write_and_read_text(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_text("a/b.txt", "hello")
        assert s.read_text("a/b.txt") == "hello"

    def test_write_and_read_bytes(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_bytes("bin.dat", b"\x00\x01\x02")
        assert s.read_bytes("bin.dat") == b"\x00\x01\x02"

    def test_read_missing_raises(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            s.read_text("nope.txt")

    def test_exists(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        assert not s.exists("x.txt")
        s.write_text("x.txt", "hi")
        assert s.exists("x.txt")

    def test_delete(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_text("del.txt", "bye")
        s.delete("del.txt")
        assert not s.exists("del.txt")

    def test_delete_missing_is_noop(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.delete("missing.txt")

    def test_list_keys(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_text("run/a.py", "a")
        s.write_text("run/sub/b.py", "b")
        s.write_text("other/c.py", "c")
        keys = s.list_keys("run")
        assert "run/a.py" in keys
        assert "run/sub/b.py" in keys
        assert "other/c.py" not in keys

    def test_list_keys_empty(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        assert s.list_keys("nonexistent") == []

    def test_delete_prefix(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_text("run/a.py", "a")
        s.write_text("run/b.py", "b")
        s.delete_prefix("run")
        assert not (tmp_path / "run").exists()

    def test_download_to_local(self, tmp_path: Path):
        root = tmp_path / "storage"
        root.mkdir()
        s = LocalStorage(root)
        s.write_text("src.txt", "data")
        dest = tmp_path / "dest" / "out.txt"
        s.download_to_local("src.txt", dest)
        assert dest.read_text() == "data"

    def test_download_prefix_to_local(self, tmp_path: Path):
        root = tmp_path / "storage"
        root.mkdir()
        s = LocalStorage(root)
        s.write_text("run/a.py", "a")
        s.write_text("run/sub/b.py", "b")
        dest = tmp_path / "local"
        s.download_prefix_to_local("run", dest)
        assert (dest / "a.py").read_text() == "a"
        assert (dest / "sub" / "b.py").read_text() == "b"

    def test_upload_from_local(self, tmp_path: Path):
        root = tmp_path / "storage"
        root.mkdir()
        s = LocalStorage(root)
        local_file = tmp_path / "upload.txt"
        local_file.write_text("uploaded")
        s.upload_from_local(local_file, "dest/upload.txt")
        assert s.read_text("dest/upload.txt") == "uploaded"

    def test_sync_local_to_storage(self, tmp_path: Path):
        root = tmp_path / "storage"
        root.mkdir()
        s = LocalStorage(root)
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "x.py").write_text("x")
        (local_dir / "sub").mkdir()
        (local_dir / "sub" / "y.py").write_text("y")
        count = s.sync_local_to_storage(local_dir, "run-1")
        assert count == 2
        assert s.read_text("run-1/x.py") == "x"
        assert s.read_text("run-1/sub/y.py") == "y"

    def test_walk(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        s.write_text("run/a.py", "aaa")
        s.write_text("run/b.py", "bb")
        items = list(s.walk("run"))
        assert len(items) == 2
        names = {name for name, _ in items}
        assert names == {"a.py", "b.py"}

    def test_presign_returns_none(self, tmp_path: Path):
        s = LocalStorage(tmp_path)
        assert s.presign_url("anything") is None


# ── RunStorage ────────────────────────────────────────────────────────────────


class TestRunStorage:
    def test_four_subfolder_key_structure(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-abc-123")

        rs.write_input("meeting.docx", "doc content")
        rs.write_requirement("req-001.json", '{"title": "Login"}')
        rs.write_spec("auth-login.json", '{"capability": "auth"}')
        rs.write_output("main.py", "print('hi')")

        assert (tmp_path / "run-abc-123" / "input" / "meeting.docx").exists()
        assert (tmp_path / "run-abc-123" / "requirements" / "req-001.json").exists()
        assert (tmp_path / "run-abc-123" / "specs" / "auth-login.json").exists()
        assert (tmp_path / "run-abc-123" / "output" / "main.py").exists()

    def test_read_output(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_output("code.py", "x = 1")
        assert rs.read_output("code.py") == "x = 1"

    def test_output_exists(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        assert not rs.output_exists("nope.py")
        rs.write_output("yes.py", "y")
        assert rs.output_exists("yes.py")

    def test_write_input_bytes(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_input_bytes("data.bin", b"\xff\xfe")
        assert rs.backend.read_bytes("run-1/input/data.bin") == b"\xff\xfe"

    def test_upload_input_from_local(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        local = tmp_path / "upload" / "doc.pdf"
        local.parent.mkdir()
        local.write_bytes(b"pdf-data")
        rs.upload_input_from_local(local, "doc.pdf")
        assert backend.exists("run-1/input/doc.pdf")

    def test_sync_input_from_local(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        local_dir = tmp_path / "uploads"
        local_dir.mkdir()
        (local_dir / "a.md").write_text("# A")
        (local_dir / "b.txt").write_text("B")
        count = rs.sync_input_from_local(local_dir)
        assert count == 2
        assert backend.exists("run-1/input/a.md")
        assert backend.exists("run-1/input/b.txt")

    def test_download_input_to_local(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_input("spec.md", "# Spec")
        dest = tmp_path / "scratch"
        rs.download_input_to_local(dest)
        assert (dest / "spec.md").read_text() == "# Spec"

    def test_sync_output_from_local(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "app.py").write_text("app")
        (scratch / "tests").mkdir()
        (scratch / "tests" / "test_app.py").write_text("test")
        count = rs.sync_output_from_local(scratch)
        assert count == 2
        assert rs.read_output("app.py") == "app"
        assert rs.read_output("tests/test_app.py") == "test"

    def test_download_output_to_local(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_output("main.py", "main")
        rs.write_output("lib/util.py", "util")
        dest = tmp_path / "download"
        rs.download_output_to_local(dest)
        assert (dest / "main.py").read_text() == "main"
        assert (dest / "lib" / "util.py").read_text() == "util"

    def test_list_input_and_output(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_input("a.md", "a")
        rs.write_output("b.py", "b")
        inputs = rs.list_input()
        outputs = rs.list_output()
        assert any("a.md" in k for k in inputs)
        assert any("b.py" in k for k in outputs)

    def test_walk_output(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_output("x.py", "xxx")
        rs.write_output("y.py", "yy")
        items = list(rs.walk_output())
        names = {name for name, _ in items}
        assert names == {"x.py", "y.py"}

    def test_delete_run(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_input("a.md", "a")
        rs.write_output("b.py", "b")
        rs.delete_run()
        assert not (tmp_path / "run-1").exists()

    def test_presign_output(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        # LocalStorage returns None for presign
        assert rs.presign_output("file.py") is None

    def test_requirements_read_write(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_requirement("req-abc.json", '{"title": "Login"}')
        assert rs.read_requirement("req-abc.json") == '{"title": "Login"}'

    def test_requirements_list(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_requirement("r1.json", "{}")
        rs.write_requirement("r2.json", "{}")
        keys = rs.list_requirements()
        assert len(keys) == 2
        assert any("r1.json" in k for k in keys)

    def test_requirements_sync_and_download(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        local = tmp_path / "reqs"
        local.mkdir()
        (local / "a.json").write_text('{"id": "a"}')
        rs.sync_requirements_from_local(local)
        dest = tmp_path / "dl"
        rs.download_requirements_to_local(dest)
        assert (dest / "a.json").read_text() == '{"id": "a"}'

    def test_walk_requirements(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_requirement("r1.json", "x")
        items = list(rs.walk_requirements())
        assert len(items) == 1
        assert items[0][0] == "r1.json"

    def test_specs_read_write(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_spec("auth.json", '{"capability": "auth"}')
        assert rs.read_spec("auth.json") == '{"capability": "auth"}'

    def test_specs_list(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_spec("s1.json", "{}")
        rs.write_spec("s2.json", "{}")
        keys = rs.list_specs()
        assert len(keys) == 2

    def test_specs_sync_and_download(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        local = tmp_path / "specs"
        local.mkdir()
        (local / "auth.json").write_text('{"id": "auth"}')
        rs.sync_specs_from_local(local)
        dest = tmp_path / "dl"
        rs.download_specs_to_local(dest)
        assert (dest / "auth.json").read_text() == '{"id": "auth"}'

    def test_walk_specs(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_spec("s1.json", "x")
        items = list(rs.walk_specs())
        assert len(items) == 1
        assert items[0][0] == "s1.json"

    def test_delete_run_clears_all_subfolders(self, tmp_path: Path):
        backend = LocalStorage(tmp_path)
        rs = RunStorage(backend, "run-1")
        rs.write_input("a.md", "a")
        rs.write_requirement("r.json", "{}")
        rs.write_spec("s.json", "{}")
        rs.write_output("b.py", "b")
        rs.delete_run()
        assert not (tmp_path / "run-1").exists()

    def test_prefixes(self):
        backend = MagicMock()
        rs = RunStorage(backend, "run-abc")
        assert rs.input_prefix == "run-abc/input"
        assert rs.requirements_prefix == "run-abc/requirements"
        assert rs.specs_prefix == "run-abc/specs"
        assert rs.output_prefix == "run-abc/output"
        assert rs.run_prefix == "run-abc"


# ── S3Storage (mocked boto3) ──────────────────────────────────────────────────


class TestS3Storage:
    def _make_storage(self) -> tuple[S3Storage, MagicMock]:
        """Create an S3Storage with mocked client."""
        s = S3Storage.__new__(S3Storage)
        s.bucket = "dark-factory-bucket"
        mock_client = MagicMock()
        mock_client.exceptions = MagicMock()
        s._client = mock_client
        s._botocore_error = Exception  # stand-in for botocore.exceptions.ClientError
        return s, mock_client

    def test_write_bytes(self):
        s, client = self._make_storage()
        s.write_bytes("run-1/output/main.py", b"code")
        client.put_object.assert_called_once_with(
            Bucket="dark-factory-bucket",
            Key="run-1/output/main.py",
            Body=b"code",
        )

    def test_write_text(self):
        s, client = self._make_storage()
        s.write_text("run-1/input/spec.md", "# Spec")
        client.put_object.assert_called_once_with(
            Bucket="dark-factory-bucket",
            Key="run-1/input/spec.md",
            Body=b"# Spec",
        )

    def test_read_bytes(self):
        s, client = self._make_storage()
        body = MagicMock()
        body.read.return_value = b"content"
        client.get_object.return_value = {"Body": body}
        assert s.read_bytes("run-1/output/f.py") == b"content"
        client.get_object.assert_called_once_with(
            Bucket="dark-factory-bucket", Key="run-1/output/f.py"
        )

    def test_exists_true(self):
        s, client = self._make_storage()
        client.head_object.return_value = {}
        assert s.exists("run-1/output/f.py") is True

    def test_exists_false(self):
        s, client = self._make_storage()
        client.head_object.side_effect = Exception("404")
        assert s.exists("run-1/output/f.py") is False

    def test_delete(self):
        s, client = self._make_storage()
        s.delete("run-1/output/f.py")
        client.delete_object.assert_called_once_with(
            Bucket="dark-factory-bucket", Key="run-1/output/f.py"
        )

    def test_list_keys(self):
        s, client = self._make_storage()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "run-1/output/a.py"},
                    {"Key": "run-1/output/b.py"},
                ]
            }
        ]
        keys = s.list_keys("run-1/output")
        assert keys == ["run-1/output/a.py", "run-1/output/b.py"]

    def test_delete_prefix(self):
        s, client = self._make_storage()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "run-1/input/a.md"},
                    {"Key": "run-1/output/b.py"},
                ]
            }
        ]
        s.delete_prefix("run-1")
        client.delete_objects.assert_called_once()
        call_args = client.delete_objects.call_args
        deleted_keys = {o["Key"] for o in call_args[1]["Delete"]["Objects"]}
        assert deleted_keys == {"run-1/input/a.md", "run-1/output/b.py"}

    def test_presign_url(self):
        s, client = self._make_storage()
        client.generate_presigned_url.return_value = "https://s3.example.com/signed"
        url = s.presign_url("run-1/output/report.html", expires=600)
        assert url == "https://s3.example.com/signed"

    def test_walk(self):
        s, client = self._make_storage()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "run-1/output/a.py", "Size": 100},
                    {"Key": "run-1/output/sub/b.py", "Size": 200},
                ]
            }
        ]
        items = list(s.walk("run-1/output"))
        assert ("a.py", 100) in items
        assert ("sub/b.py", 200) in items


# ── S3 + RunStorage integration ──────────────────────────────────────────────


class TestS3RunStorage:
    def test_run_storage_keys_are_correct(self):
        """Verify RunStorage constructs the right S3 keys for all four subfolders."""
        s, client = TestS3Storage()._make_storage()
        rs = RunStorage(s, "run-20260412-051047-9677")

        rs.write_input("meeting.docx", "doc")
        client.put_object.assert_called_with(
            Bucket="dark-factory-bucket",
            Key="run-20260412-051047-9677/input/meeting.docx",
            Body=b"doc",
        )

        client.reset_mock()
        rs.write_requirement("req-001.json", '{"title":"Login"}')
        client.put_object.assert_called_with(
            Bucket="dark-factory-bucket",
            Key="run-20260412-051047-9677/requirements/req-001.json",
            Body=b'{"title":"Login"}',
        )

        client.reset_mock()
        rs.write_spec("auth.json", '{"capability":"auth"}')
        client.put_object.assert_called_with(
            Bucket="dark-factory-bucket",
            Key="run-20260412-051047-9677/specs/auth.json",
            Body=b'{"capability":"auth"}',
        )

        client.reset_mock()
        rs.write_output("main.py", "code")
        client.put_object.assert_called_with(
            Bucket="dark-factory-bucket",
            Key="run-20260412-051047-9677/output/main.py",
            Body=b"code",
        )

    def test_delete_run_uses_run_prefix(self):
        s, client = TestS3Storage()._make_storage()
        rs = RunStorage(s, "run-abc")
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "run-abc/input/a.md"},
                    {"Key": "run-abc/output/b.py"},
                ]
            }
        ]
        rs.delete_run()
        # Should delete with prefix "run-abc/"
        client.delete_objects.assert_called_once()


# ── Factory ───────────────────────────────────────────────────────────────────


class TestGetStorage:
    def setup_method(self):
        reset_storage()

    def teardown_method(self):
        reset_storage()

    def test_default_is_local(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STORAGE_BACKEND", None)
            s = get_storage()
            assert isinstance(s, LocalStorage)

    def test_explicit_local(self):
        with patch.dict(os.environ, {"STORAGE_BACKEND": "local"}):
            s = get_storage()
            assert isinstance(s, LocalStorage)

    def test_s3_requires_bucket(self):
        with patch.dict(os.environ, {"STORAGE_BACKEND": "s3"}):
            os.environ.pop("S3_BUCKET", None)
            with pytest.raises(ValueError, match="S3_BUCKET"):
                get_storage()

    def test_singleton(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STORAGE_BACKEND", None)
            a = get_storage()
            b = get_storage()
            assert a is b
