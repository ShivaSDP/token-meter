import hashlib
import io
import json
import os
import plistlib
import tarfile
import tempfile
import unittest
from unittest import mock

from token_meter.services import work_setup as S


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Run:
    def __init__(self, returncode=0, stderr=""):
        self.returncode, self.stderr, self.stdout = returncode, stderr, ""


def git_digest(data):
    return "git:" + hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class WorkSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = {"model": "token-meter-jet", "ollama_url": "http://127.0.0.1:11434"}
        self.urls, self.commands = [], []
        self.ollama = {}  # url -> {"version":..., "models": [...]}
        self.files = {}

    def opener(self, request, timeout=0):
        url = request.full_url
        self.urls.append(url)
        for base, info in self.ollama.items():
            if url == base + "/api/version":
                return FakeResponse(json.dumps({"version": info["version"]}).encode())
            if url == base + "/api/tags":
                return FakeResponse(json.dumps({"models": [{"name": n} for n in info["models"]]}).encode())
        if url in self.files:
            return FakeResponse(self.files[url])
        raise OSError("unreachable")

    def runner(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == "/usr/bin/codesign":
            return Run(0, f"TeamIdentifier={S.OLLAMA_TEAM_ID}\n")
        if command[:2] == ["/bin/launchctl", "bootstrap"]:
            self.ollama[S.MANAGED_URL] = {"version": S.OLLAMA_VERSION, "models": []}
        if len(command) > 1 and command[1] == "create":
            target = "http://" + kwargs["env"]["OLLAMA_HOST"]
            self.ollama[target]["models"].append(command[2] + ":latest")
        return Run()

    def make(self, **kwargs):
        def set_url(url):
            self.settings["ollama_url"] = url
        return S.WorkSetup(base_dir=os.path.join(self.tmp.name, "ollama"), cache_dir=os.path.join(self.tmp.name, "cache"),
                           launch_agents_dir=os.path.join(self.tmp.name, "agents"), get_settings=lambda: dict(self.settings),
                           set_ollama_url=set_url, opener=self.opener, runner=self.runner, uid=501,
                           sleep=lambda s: None, disk_free=kwargs.pop("disk_free", lambda p: 10 ** 13), **kwargs)

    def small_model(self):
        files = []
        for name in ("config.json", "model-00001-of-00001.safetensors"):
            data = name.encode() * 3
            digest = git_digest(data) if name.endswith(".json") else "sha256:" + hashlib.sha256(data).hexdigest()
            files.append((name, len(data), digest))
            self.files[f"{S.JET_BASE_URL}/{name}"] = data
        return mock.patch.object(S, "JET_FILES", tuple(files))

    def ollama_archive(self, extra=None):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
            for name, data in [("ollama", b"\xcf\xfa\xed\xfe" + b"x" * 20), *(extra or [])]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        data = buffer.getvalue()
        self.files[S.OLLAMA_ARCHIVE_URL] = data
        return mock.patch.multiple(S, OLLAMA_ARCHIVE_SIZE=len(data),
                                   OLLAMA_ARCHIVE_DIGEST="sha256:" + hashlib.sha256(data).hexdigest())

    def test_existing_ollama_with_the_model_needs_nothing(self):
        self.ollama["http://127.0.0.1:11434"] = {"version": "0.34.4", "models": ["token-meter-jet:latest"]}
        ready = mock.Mock()
        with mock.patch.object(S, "CLI_CANDIDATES", (self.runner_path(),)):
            setup = self.make(on_ready=ready)
            setup.run()
        self.assertEqual(setup.status()["state"], "ready")
        ready.assert_called_once()
        self.assertFalse(any("huggingface" in u or "github" in u for u in self.urls))

    def runner_path(self):
        path = os.path.join(self.tmp.name, "ollama-cli")
        with open(path, "w") as handle:
            handle.write("#!/bin/sh\n")
        os.chmod(path, 0o755)
        return path

    def test_missing_ollama_installs_the_pinned_build_and_imports_the_model(self):
        with self.ollama_archive(), self.small_model(), mock.patch.object(S, "CLI_CANDIDATES", ()):
            setup = self.make()
            setup.run()
        self.assertEqual(self.settings["ollama_url"], S.MANAGED_URL)
        self.assertTrue(os.path.isfile(setup.binary))
        with open(setup.plist_path, "rb") as handle:
            plist = plistlib.load(handle)
        self.assertEqual(plist["ProgramArguments"], [setup.binary, "serve"])
        self.assertEqual(plist["EnvironmentVariables"]["OLLAMA_HOST"], "127.0.0.1:11435")
        self.assertEqual(plist["StandardErrorPath"], "/dev/null")
        create = next(c for c in self.commands if len(c) > 1 and c[1] == "create")
        self.assertEqual(create[:5], [setup.binary, "create", "token-meter-jet", "-q", "int4"])
        self.assertIn("token-meter-jet:latest", self.ollama[S.MANAGED_URL]["models"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "cache", f"jet-{S.JET_REVISION}")))
        self.assertTrue(all(not c[0].endswith("sudo") for c in self.commands))

    def test_tampered_download_is_rejected_and_removed(self):
        with self.ollama_archive(), mock.patch.object(S, "CLI_CANDIDATES", ()):
            self.files[S.OLLAMA_ARCHIVE_URL] = b"y" * S.OLLAMA_ARCHIVE_SIZE
            setup = self.make()
            with self.assertRaises(S.SetupError) as caught:
                setup.run()
        self.assertEqual(caught.exception.reason, "verify")
        self.assertFalse(os.path.exists(setup.binary))
        self.assertFalse(os.listdir(os.path.join(self.tmp.name, "cache")))

    def test_unsigned_binary_is_rejected(self):
        def runner(command, **kwargs):
            if command[0] == "/usr/bin/codesign":
                return Run(0, "TeamIdentifier=SOMEONEELSE\n")
            return self.runner(command, **kwargs)
        with self.ollama_archive(), mock.patch.object(S, "CLI_CANDIDATES", ()):
            setup = self.make()
            setup.runner = runner
            with self.assertRaises(S.SetupError) as caught:
                setup.run()
        self.assertEqual(caught.exception.reason, "verify")
        self.assertFalse(os.path.exists(setup.binary))

    def test_archive_paths_outside_the_folder_are_refused(self):
        with self.ollama_archive(extra=[("../escape", b"x")]), mock.patch.object(S, "CLI_CANDIDATES", ()):
            setup = self.make()
            with self.assertRaises(S.SetupError):
                setup.run()
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "ollama", "escape")))

    def test_low_disk_space_stops_before_downloading(self):
        self.ollama["http://127.0.0.1:11434"] = {"version": "0.34.4", "models": []}
        with self.small_model(), mock.patch.object(S, "CLI_CANDIDATES", (self.runner_path(),)):
            setup = self.make(disk_free=lambda p: 1)
            setup._run_safely()
        status = setup.status()
        self.assertEqual((status["state"], status["reason"]), ("failed", "disk_space"))
        self.assertGreater(status["needed_bytes"], 0)
        self.assertFalse(any("huggingface" in u for u in self.urls))

    def test_old_ollama_is_replaced_by_the_managed_runtime(self):
        self.ollama["http://127.0.0.1:11434"] = {"version": "0.20.1", "models": []}
        with self.ollama_archive(), self.small_model(), mock.patch.object(S, "CLI_CANDIDATES", ()):
            self.make().run()
        self.assertEqual(self.settings["ollama_url"], S.MANAGED_URL)

    def test_stop_agent_unloads_and_removes_the_launch_agent(self):
        with self.ollama_archive(), self.small_model(), mock.patch.object(S, "CLI_CANDIDATES", ()):
            setup = self.make()
            setup.run()
        self.assertTrue(setup.stop_agent())
        self.assertFalse(os.path.exists(setup.plist_path))
        self.assertIn(["/bin/launchctl", "bootout", f"gui/501/{S.AGENT_LABEL}"], self.commands)
        self.assertFalse(setup.stop_agent())

    def test_failures_report_only_a_reason_code(self):
        setup = self.make()
        setup.get_settings = mock.Mock(side_effect=RuntimeError("/Users/someone/secret path"))
        setup._run_safely()
        status = setup._state
        self.assertEqual((status["state"], status["reason"]), ("failed", "internal"))
        self.assertNotIn("secret", json.dumps(status))

    def test_pinned_sources_and_versions(self):
        self.assertTrue(S.OLLAMA_ARCHIVE_URL.startswith("https://github.com/ollama/ollama/releases/download/v0.34.4/"))
        self.assertTrue(S.JET_BASE_URL.endswith("/resolve/" + S.JET_COMMIT))
        self.assertEqual(len(S.JET_FILES), 16)
        self.assertEqual(S.MANAGED_URL, "http://127.0.0.1:11435")
        self.assertTrue(all(d.startswith(("sha256:", "git:")) for _p, _s, d in S.JET_FILES))


if __name__ == "__main__":
    unittest.main()
