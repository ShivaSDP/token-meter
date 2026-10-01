"""Hands-off Work insights setup on macOS: a pinned Ollama runtime and the pinned Jet model.

Every download comes from a pinned URL and is checked against a pinned size and hash before use.
Nothing needs administrator rights: Ollama lives under Token Meter's Application Support folder and
runs as a per-user LaunchAgent bound to loopback.
"""

import argparse
import hashlib
import http.client
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

OLLAMA_VERSION = "0.34.4"
OLLAMA_ARCHIVE_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-darwin.tgz"
OLLAMA_ARCHIVE_SIZE = 160_042_307
OLLAMA_ARCHIVE_DIGEST = "sha256:e9c8fddaab5f48f47f2c4ae3d23d0732f5182417125353faeed2188e34a22799"
OLLAMA_TEAM_ID = "3MU9H2V9Y9"
MIN_OLLAMA_VERSION = (0, 34, 0)
MANAGED_PORT = 11435
MANAGED_URL = f"http://127.0.0.1:{MANAGED_PORT}"
AGENT_LABEL = "com.token-meter.ollama"

JET_REVISION = "v6.2.0"
JET_COMMIT = "fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a"
JET_BASE_URL = f"https://huggingface.co/michaljach/jet/resolve/{JET_COMMIT}"
JET_FILES = (
    ("LICENSE", 11544, "git:f938136e3adacfd92be087f6e113b5d6d97f678f"),
    ("calibration.json", 95, "git:46664e3021f440b46656ca88d8ef9b1b9efd8505"),
    ("chat_template.jinja", 7756, "git:a585dec894e63da457d9440ec6aa7caa16d20860"),
    ("config.json", 1979, "git:fb849c71825b3d42be90bc7a5a52438f9143118d"),
    ("model-00001-of-00009.safetensors", 1271398544, "sha256:758c65ea6c124bbe96b68a2d5475f7e4ec0ff9b6a4dbba87f295144ace55f07d"),
    ("model-00002-of-00009.safetensors", 990906704, "sha256:8a79a65e7b3187a1545178737c3afb68c5563da8ef08a843ac0d804d50f13126"),
    ("model-00003-of-00009.safetensors", 991305848, "sha256:cbdcdbab53b2e604805ed88324f27ecc1eb4f3abf105993820a0818399524436"),
    ("model-00004-of-00009.safetensors", 990906712, "sha256:43db7601b74bfc88b9c2cc07d56d52de1a9f9ff64a987b84d311375530320ba5"),
    ("model-00005-of-00009.safetensors", 985663824, "sha256:4997734d89a1b29dc6f725faf013f1fbeaf21af78893e9fb4e0f7fd16b5f1293"),
    ("model-00006-of-00009.safetensors", 983687224, "sha256:b1a42ff788a08003b053f50ca7aec1e87159f03a4d7d83bc1faf2d0e6dd05ccf"),
    ("model-00007-of-00009.safetensors", 999026496, "sha256:077cc7ee8901098fd751bd457dc18a059d4ee27af50f491f83ef21fd7aaf8179"),
    ("model-00008-of-00009.safetensors", 982888776, "sha256:1cddde32313f1a4beafa864a0f89eafa92e61b248fdecda83910ea070a613471"),
    ("model-00009-of-00009.safetensors", 215774320, "sha256:83274a5302791056a5ca1ded5fce9ada6659bc760d34835a11bd89a01f31846d"),
    ("model.safetensors.index.json", 35567, "git:5ac40985a7b36ab61092e6cac05d91b6f01291cf"),
    ("tokenizer.json", 19989325, "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ("tokenizer_config.json", 1125, "git:ed1f99f3d50eecc9850623be644bd1131686cbe1"),
)
# `ollama create` copies the weights into its store before writing the 4-bit model.
IMPORT_RESERVE_BYTES = sum(size for _path, size, _digest in JET_FILES) + 3 * 1024 ** 3
CHUNK = 1 << 20
CLI_CANDIDATES = ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
                  "/Applications/Ollama.app/Contents/Resources/ollama")
# An installed Ollama that is not answering yet (for example at login) gets this long to come up.
OWN_OLLAMA_WAIT_S = 120
MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}

IDLE, CHECKING, INSTALLING, STARTING, DOWNLOADING, IMPORTING, READY, FAILED = (
    "idle", "checking", "installing_ollama", "starting_ollama", "downloading_model", "importing", "ready", "failed")
REASONS = ("network", "verify", "disk_space", "ollama_start", "ollama_offline", "import", "cancelled", "internal")
# Loopback probes must never go through a system or environment HTTP proxy.
_LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SetupError(Exception):
    def __init__(self, reason, needed_bytes=0):
        super().__init__(reason)
        self.reason = reason if reason in REASONS else "internal"
        self.needed_bytes = needed_bytes


def file_digest(path, kind):
    """``sha256:<hex>`` for LFS weights, ``git:<hex>`` (git blob SHA-1) for small tracked files."""
    if kind == "sha256":
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return f"{'sha256' if kind == 'sha256' else 'git'}:{digest.hexdigest()}"


def verified(path, size, digest):
    try:
        return os.path.getsize(path) == size and file_digest(path, digest.split(":", 1)[0]) == digest
    except OSError:
        return False


def parse_version(text):
    parts = []
    for piece in str(text or "").split("-")[0].split(".")[:3]:
        if not piece.isdigit():
            return None
        parts.append(int(piece))
    return tuple(parts + [0] * (3 - len(parts))) if parts else None


def safe_extract(archive, destination):
    """Extract regular files, directories, and same-folder symlinks; nothing may land outside ``destination``."""
    root = os.path.realpath(destination)

    def inside(path):
        return path == root or path.startswith(root + os.sep)

    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            name = os.path.normpath(member.name)
            if os.path.isabs(name) or name == ".." or name.startswith(".." + os.sep):
                raise SetupError("verify")
            target = os.path.join(root, name)
            parent = os.path.realpath(os.path.dirname(target))
            if not inside(parent) or os.path.islink(target):
                raise SetupError("verify")
            if member.isdir():
                os.makedirs(target, mode=0o755, exist_ok=True)
            elif member.issym():
                # Library aliases point at a sibling file; anything else could chain out of the folder.
                if os.sep in member.linkname or member.linkname in ("", ".", ".."):
                    raise SetupError("verify")
                os.symlink(member.linkname, target)
            elif member.isfile():
                os.makedirs(os.path.dirname(target), mode=0o755, exist_ok=True)
                source = bundle.extractfile(member)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     member.mode & 0o755)
                with os.fdopen(descriptor, "wb") as handle, source:
                    shutil.copyfileobj(source, handle, CHUNK)
            else:
                raise SetupError("verify")


class WorkSetup:
    """One background setup job; ``status()`` is bounded, content-free, and safe to serve."""

    def __init__(self, *, base_dir, cache_dir, launch_agents_dir, get_settings, set_ollama_url,
                 on_ready=lambda: None, opener=None, runner=subprocess.run, uid=None, sleep=time.sleep,
                 disk_free=None, log=None):
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.plist_path = os.path.join(launch_agents_dir, AGENT_LABEL + ".plist")
        self.get_settings = get_settings
        self.set_ollama_url = set_ollama_url
        self.on_ready = on_ready
        self.opener = opener or urllib.request.urlopen
        self.loopback_opener = opener or _LOOPBACK_OPENER.open
        self.runner = runner
        self.uid = os.getuid() if uid is None else uid
        self.sleep = sleep
        self.disk_free = disk_free or (lambda path: shutil.disk_usage(path).free)
        self.log = log or (lambda message: None)
        self._lock = threading.Lock()
        self._thread = None
        self._cancel = threading.Event()
        self._state = {"state": IDLE, "reason": "", "done_bytes": 0, "total_bytes": 0, "needed_bytes": 0}

    @property
    def binary(self):
        return os.path.join(self.base_dir, OLLAMA_VERSION, "ollama")

    @property
    def models_dir(self):
        return os.path.join(self.base_dir, "models")

    def status(self):
        with self._lock:
            return dict(self._state, managed=self.get_settings().get("ollama_url") == MANAGED_URL,
                        ollama_version=OLLAMA_VERSION)

    def _set(self, state, **values):
        with self._lock:
            self._state = {"state": state, "reason": "", "done_bytes": 0, "total_bytes": 0, "needed_bytes": 0,
                           **values}
        self.log(state)

    def _progress(self, done):
        with self._lock:
            self._state["done_bytes"] = done

    def start(self):
        """Run setup in the background unless it is already running; returns True when started."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel.clear()
            self._state = {"state": CHECKING, "reason": "", "done_bytes": 0, "total_bytes": 0, "needed_bytes": 0}
            self._thread = threading.Thread(target=self._run_safely, name="work-setup", daemon=True)
            self._thread.start()
            return True

    def cancel(self, wait_s=10.0):
        """Stop a running setup at its next checkpoint, waiting briefly for it to finish."""
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(wait_s)

    def _checkpoint(self):
        if self._cancel.is_set():
            raise SetupError("cancelled")

    def _run_safely(self):
        try:
            self.run()
        except SetupError as error:
            if error.reason == "cancelled":
                self._set(IDLE)
            else:
                self._set(FAILED, reason=error.reason, needed_bytes=error.needed_bytes)
        except Exception:
            self._set(FAILED, reason="internal")

    def run(self):
        settings = self.get_settings()
        self._set(CHECKING)
        url, cli = self._ollama(settings["ollama_url"])
        has_model = self._has_model(url, settings["model"])
        if has_model is None:
            raise SetupError("ollama_start")
        if not has_model:
            folder = self._download_model()
            self._checkpoint()
            self._set(IMPORTING)
            self._import(cli, url, settings["model"], folder)
            shutil.rmtree(folder, ignore_errors=True)
        self._set(READY)
        self.on_ready()

    # Ollama runtime

    def _get_json(self, url, timeout=3):
        try:
            with self.loopback_opener(urllib.request.Request(url), timeout=timeout) as response:
                return json.loads(response.read(65536).decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
            return None

    def _version(self, url):
        payload = self._get_json(url + "/api/version")
        return parse_version(payload.get("version")) if isinstance(payload, dict) else None

    def _has_model(self, url, model):
        """True or False from Ollama's model list; None when Ollama did not answer."""
        payload = self._get_json(url + "/api/tags", timeout=20)
        if not isinstance(payload, dict):
            return None
        names = {str(item.get("name") or "") for item in payload.get("models") or [] if isinstance(item, dict)}
        return model in names or f"{model}:latest" in names

    def _ollama(self, configured):
        version = self._version(configured)
        if configured != MANAGED_URL and version is None and self._find_cli():
            # The user's own Ollama is installed but not answering yet; wait for it rather than replace it.
            for _attempt in range(int(OWN_OLLAMA_WAIT_S / 2)):
                self._checkpoint()
                self.sleep(2)
                version = self._version(configured)
                if version is not None:
                    break
            else:
                raise SetupError("ollama_offline")
        if configured != MANAGED_URL and version and version >= MIN_OLLAMA_VERSION:
            return configured, self._find_cli() or self._install_ollama()
        binary = self._install_ollama()
        self._checkpoint()
        if not self._version(MANAGED_URL):
            self._set(STARTING)
            self.start_agent(binary)
            for _attempt in range(60):
                if self._version(MANAGED_URL):
                    break
                if self._cancel.is_set():
                    self.stop_agent()
                    raise SetupError("cancelled")
                self.sleep(0.5)
            else:
                self.stop_agent()
                raise SetupError("ollama_start")
        self._checkpoint()
        if configured != MANAGED_URL:
            self.set_ollama_url(MANAGED_URL)
        return MANAGED_URL, binary

    def _find_cli(self):
        for path in CLI_CANDIDATES:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _signed_by_ollama(self, path):
        try:
            check = self.runner(["/usr/bin/codesign", "--verify", "--strict", path],
                                capture_output=True, text=True, timeout=30)
            detail = self.runner(["/usr/bin/codesign", "-dv", path], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        return check.returncode == 0 and f"TeamIdentifier={OLLAMA_TEAM_ID}" in (detail.stderr or "")

    def _verify_bundle(self, folder):
        for directory, _dirs, files in os.walk(folder):
            for name in files:
                path = os.path.join(directory, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                with open(path, "rb") as handle:
                    magic = handle.read(4)
                if magic in MACHO_MAGIC and not self._signed_by_ollama(path):
                    raise SetupError("verify")

    def _install_ollama(self):
        final = os.path.dirname(self.binary)
        if os.path.isfile(self.binary) and self._signed_by_ollama(self.binary):
            return self.binary
        self._set(INSTALLING, total_bytes=OLLAMA_ARCHIVE_SIZE)
        os.makedirs(self.cache_dir, exist_ok=True)
        archive = os.path.join(self.cache_dir, f"ollama-{OLLAMA_VERSION}-darwin.tgz")
        self._download(OLLAMA_ARCHIVE_URL, archive, OLLAMA_ARCHIVE_SIZE, OLLAMA_ARCHIVE_DIGEST, 0)
        partial = final + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        os.makedirs(partial)
        try:
            safe_extract(archive, partial)
            self._verify_bundle(partial)
        except (tarfile.TarError, OSError, SetupError):
            shutil.rmtree(partial, ignore_errors=True)
            os.remove(archive)  # A retry downloads it again rather than repeating the same failure.
            raise SetupError("verify") from None
        shutil.rmtree(final, ignore_errors=True)
        os.replace(partial, final)
        os.remove(archive)
        return self.binary

    def agent_plist(self, binary):
        return {
            "Label": AGENT_LABEL,
            "ProgramArguments": [binary, "serve"],
            "EnvironmentVariables": {
                "OLLAMA_HOST": f"127.0.0.1:{MANAGED_PORT}",
                "OLLAMA_MODELS": self.models_dir,
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_NUM_PARALLEL": "1",
            },
            "RunAtLoad": True,
            "KeepAlive": True,
            # Request logs can name models and timings; keep them out of files.
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "/dev/null",
        }

    def start_agent(self, binary):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.plist_path), exist_ok=True)
        temporary = self.plist_path + ".tmp"
        with open(temporary, "wb") as handle:
            plistlib.dump(self.agent_plist(binary), handle)
        os.replace(temporary, self.plist_path)
        domain = f"gui/{self.uid}"
        self.runner(["/bin/launchctl", "bootout", f"{domain}/{AGENT_LABEL}"], capture_output=True, timeout=30)
        for attempt in range(3):
            # launchd unloads asynchronously, so a bootstrap right after bootout can briefly fail.
            result = self.runner(["/bin/launchctl", "bootstrap", domain, self.plist_path],
                                 capture_output=True, timeout=30)
            if result.returncode == 0:
                return
            self.sleep(1.0 + attempt)
        self.stop_agent()
        raise SetupError("ollama_start")

    def stop_agent(self):
        """Stop the managed Ollama and keep it from starting at login; the model stays for next time."""
        if not os.path.exists(self.plist_path):
            return False
        self.runner(["/bin/launchctl", "bootout", f"gui/{self.uid}/{AGENT_LABEL}"], capture_output=True, timeout=30)
        os.remove(self.plist_path)
        with self._lock:
            if self._state["state"] == READY:
                self._state = dict(self._state, state=IDLE)
        return True

    # Jet model

    def _download(self, url, destination, size, digest, offset):
        """Resumable download that must match ``size`` and ``digest``; progress adds to ``offset``."""
        if verified(destination, size, digest):
            self._progress(offset + size)
            return
        have = os.path.getsize(destination) if os.path.exists(destination) else 0
        if have >= size:
            os.remove(destination)
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with self.opener(request, timeout=60) as response:
                if have and getattr(response, "status", 206) != 206:
                    have = 0
                with open(destination, "ab" if have else "wb") as handle:
                    done = have
                    for chunk in iter(lambda: response.read(CHUNK), b""):
                        self._checkpoint()
                        handle.write(chunk)
                        done += len(chunk)
                        if done > size:
                            break
                        self._progress(offset + done)
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise SetupError("network") from None
        if not verified(destination, size, digest):
            os.remove(destination)
            raise SetupError("verify")

    def _download_model(self):
        folder = os.path.join(self.cache_dir, f"jet-{JET_REVISION}")
        os.makedirs(folder, exist_ok=True)
        total = sum(size for _path, size, _digest in JET_FILES)
        have = sum(min(size, os.path.getsize(os.path.join(folder, path)))
                   for path, size, _digest in JET_FILES if os.path.exists(os.path.join(folder, path)))
        needed = total - have + IMPORT_RESERVE_BYTES
        if self.disk_free(folder) < needed:
            raise SetupError("disk_space", needed_bytes=needed)
        self._set(DOWNLOADING, total_bytes=total)
        offset = 0
        for path, size, digest in JET_FILES:
            self._checkpoint()
            self._download(f"{JET_BASE_URL}/{path}", os.path.join(folder, path), size, digest, offset)
            offset += size
        with open(os.path.join(folder, "Modelfile"), "w", encoding="utf-8") as handle:
            handle.write(f"FROM {folder}\nPARAMETER temperature 0\nPARAMETER num_predict 1\n")
        return folder

    def _import(self, cli, url, model, folder):
        env = dict(os.environ, OLLAMA_HOST=url.split("://", 1)[1])
        try:
            result = self.runner([cli, "create", model, "-q", "int4", "-f", os.path.join(folder, "Modelfile")],
                                 capture_output=True, text=True, timeout=3 * 3600, env=env)
        except (OSError, subprocess.SubprocessError):
            raise SetupError("import") from None
        if result.returncode != 0 or not self._has_model(url, model):
            raise SetupError("import")


def default_paths(home=None):
    home = home or os.path.expanduser("~")
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return {
        "base_dir": os.environ.get("TOKEN_METER_OLLAMA_DIR")
        or os.path.join(home, "Library", "Application Support", "Token Meter", "ollama"),
        "cache_dir": os.path.join(cache, "token-meter"),
        "launch_agents_dir": os.environ.get("TOKEN_METER_LAUNCH_AGENTS_DIR")
        or os.path.join(home, "Library", "LaunchAgents"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Set up Ollama and the Jet model for Token Meter Work insights.")
    parser.add_argument("--model", default="token-meter-jet")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("Work insights are available on macOS only.", file=sys.stderr)
        return 1
    chosen = {"url": args.ollama_url}
    setup = WorkSetup(**default_paths(), get_settings=lambda: {"model": args.model, "ollama_url": chosen["url"]},
                      set_ollama_url=lambda url: chosen.update(url=url),
                      log=lambda state: print(state.replace("_", " ")))
    try:
        setup.run()
    except SetupError as error:
        print(f"Setup failed: {error.reason.replace('_', ' ')}.", file=sys.stderr)
        return 1
    print(f"Done. Ollama is at {chosen['url']}; turn on Settings → Work insights in Token Meter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
