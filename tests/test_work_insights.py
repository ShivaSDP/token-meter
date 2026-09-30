import http.client
import http.server
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import meter
from token_meter.domain import work as domain
from token_meter.services import work_insights as W


SECRET_TEXT = "please refactor the zebra-kumquat billing module"


def letter_for(prompt, key):
    import re
    match = re.search(rf"^([A-Z]): {re.escape(key)}:", prompt, re.M)
    return match.group(1) if match else key


def jet_response(token, logprob=-0.05, others=()):
    top = [{"token": token, "logprob": logprob}] + [{"token": t, "logprob": lp} for t, lp in others]
    return {"message": {"content": token}, "logprobs": [{"token": token, "logprob": logprob, "top_logprobs": top}]}


class FakeClient:
    """Scripted stand-in for OllamaClient; records prompts it receives."""

    script = []
    responder = None
    prompts = []
    unloads = 0
    digest_error = None

    def __init__(self, url, model):
        self.url, self.model = url, model

    def model_digest(self):
        if FakeClient.digest_error:
            raise FakeClient.digest_error
        return "digest-1"

    def classify(self, prompt, timeout):
        FakeClient.prompts.append(prompt)
        if FakeClient.responder is not None:
            action = FakeClient.responder(prompt)
        else:
            action = FakeClient.script.pop(0) if FakeClient.script else jet_response("A")
        if isinstance(action, Exception):
            raise action
        return action

    def unload(self):
        FakeClient.unloads += 1


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make_service(tmp, settings=None, **kwargs):
    FakeClient.script, FakeClient.prompts, FakeClient.unloads, FakeClient.digest_error = [], [], 0, None
    FakeClient.responder = None
    values = W.normalize_settings({"enabled": True, "backfill_days": 0, "pause_on_battery": True,
                                   **(settings or {})})
    clock = kwargs.pop("clock", Clock())

    def sleep(seconds):
        clock.now += seconds

    service = W.WorkInsightsService(
        os.path.join(tmp, "work.sqlite3"), lambda: values, client_factory=FakeClient,
        clock=clock, monotonic=clock, sleep=sleep,
        load_probe=kwargs.pop("load_probe", lambda: 0.1),
        power_probe=kwargs.pop("power_probe", lambda: "ac"), **kwargs)
    return service, values, clock


def turns(*texts, context="Done: I changed the chart."):
    return [{"ts": 1_799_999_000.0 + i, "text": t, "model": "m", "context": context if i else ""}
            for i, t in enumerate(texts)]


def drain(service, limit=200):
    clock = service.clock
    for _ in range(limit):
        wait = service.step()
        if not service.queue and wait > 0:
            break
        if hasattr(clock, "now"):
            clock.now += max(wait, 0.0) + 0.01


class TextPreparationTests(unittest.TestCase):
    def test_strips_runtime_wrappers_and_citations(self):
        raw = ("# Files mentioned by the user:\n## a.png: /x.png\n## My request: fix the chart "
               "<image name=[Image #1] path=\"/x.png\"></image>")
        self.assertEqual(W.prepare_text(raw), "fix the chart")
        self.assertEqual(W.clean_text("ok <oai-mem-citation>MEMORY.md:1</oai-mem-citation>"), "ok")

    def test_skips_injected_messages(self):
        for text in ("<task-notification>done</task-notification>", "# AGENTS.md instructions",
                     "<environment_context>cwd</environment_context>"):
            self.assertEqual(W.prepare_text(text), "")

    def test_skeleton_bounds_long_text_and_keeps_ends(self):
        text = "GOAL: build the report\n" + "\n".join(f"- item {i}" for i in range(400)) + \
               "\n" + "\n".join(f"    at frame{i} (file.py:{i})" for i in range(300)) + "\nFINAL ASK: ship it"
        out = W.skeleton(text)
        self.assertLessEqual(len(out), W.MAX_ITEM_CHARS)
        self.assertTrue(out.startswith("GOAL: build the report"))
        self.assertTrue(out.endswith("FINAL ASK: ship it"))
        self.assertIn("lines omitted", out)
        self.assertEqual(W.skeleton("short"), "short")


class SettingsValidationTests(unittest.TestCase):
    def test_ollama_url_must_be_loopback_http(self):
        self.assertEqual(W.validate_ollama_url("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")
        self.assertEqual(W.validate_ollama_url("http://localhost"), "http://127.0.0.1:11434")
        self.assertEqual(W.validate_ollama_url("http://[::1]:9000"), "http://[::1]:9000")
        for bad in ("https://127.0.0.1:11434", "http://10.0.0.5:11434", "http://example.com",
                    "http://user:pw@127.0.0.1", "http://127.0.0.1/api", "file:///tmp/x"):
            with self.assertRaises(ValueError):
                W.validate_ollama_url(bad)

    def test_normalize_settings_keeps_defaults_for_invalid_fields(self):
        settings = W.normalize_settings({"enabled": "yes", "rate_per_minute": 999,
                                         "ollama_url": "http://evil.example", "areas": [{"name": "x"}]})
        self.assertEqual(settings, W.default_settings())

    def test_areas_bounds_and_reserved_names(self):
        with self.assertRaises(ValueError):
            W.normalize_areas([{"name": "One", "description": "d"}])
        with self.assertRaises(ValueError):
            W.normalize_areas([{"name": "Unclear", "description": "d"}, {"name": "B", "description": "d"}])
        with self.assertRaises(ValueError):
            W.normalize_areas([{"name": "A", "description": "d"}, {"name": "a", "description": "d"}])
        self.assertEqual(len(W.normalize_areas(list(W.DEFAULT_AREAS))), len(W.DEFAULT_AREAS))


class PromptAndReadoutTests(unittest.TestCase):
    def test_choice_prompt_uses_jet_format(self):
        prompt, labels, keys = W.render_prompt("User's message:\nhi", W.question_for("work_type", W.default_settings()))
        self.assertTrue(prompt.startswith("<state>\nUser's message:\nhi\n</state>"))
        self.assertIn("A: debug:", prompt)
        self.assertEqual(keys[0], "debug")
        self.assertEqual(labels[:2], ["A", "B"])

    def test_readout_softmaxes_label_tokens_only(self):
        _, labels, keys = W.render_prompt("s", W.question_for("work_type", W.default_settings()))
        distribution = W.read_distribution(
            jet_response("A", -0.1, [("B", -2.5), ("Hello", -0.01)]), labels, keys, 1.0)
        self.assertEqual(set(distribution), {"debug", "feature"})
        self.assertGreater(distribution["debug"], 0.8)

    def test_choice_answers_average_both_option_orders(self):
        question = W.question_for("work_type", W.default_settings())
        forward, backward = W.render_prompt("s", question), W.render_prompt("s", question, reverse=True)
        self.assertEqual(backward[2][0], "other")
        value, confidence = W.read_answer(
            [jet_response("A", -0.1, [("B", -1.0)]), jet_response(letter_for(backward[0], "feature"), -0.1,
                                                                     [(letter_for(backward[0], "debug"), -3.0)])],
            question, [(forward[1], forward[2]), (backward[1], backward[2])])
        self.assertEqual(value, "feature")
        self.assertGreater(confidence, 0.55)
        self.assertLess(confidence, 0.7)

    def test_noul_and_score_readouts(self):
        question = W.question_for("correction", W.default_settings())
        _, labels, keys = W.render_prompt("s", question)
        self.assertEqual(W.read_answer([jet_response("Yes", -0.2, [("no", -3)])], question, [(labels, keys)])[0], True)
        question = W.question_for("complexity", W.default_settings())
        _, labels, keys = W.render_prompt("s", question)
        self.assertEqual(W.read_answer([jet_response("2")], question, [(labels, keys)])[0], 2)
        level, _ = W.read_answer([jet_response("0", -1.2, [("3", -0.4)])], question, [(labels, keys)])
        self.assertEqual(level, 2)

    def test_missing_label_token_is_an_item_error(self):
        with self.assertRaises(W.ClassifierError) as caught:
            W.read_distribution(jet_response("Sure"), ["A", "B"], ["x", "y"], 1.0)
        self.assertEqual(caught.exception.kind, "item")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_disabled_service_queues_nothing(self):
        service, values, _ = make_service(self.tmp.name, {"enabled": False})
        self.assertEqual(service.observe("s1", turns("hello")), 0)
        self.assertEqual(service.step(), 5.0)
        self.assertEqual(service.status()["state"], W.STATE_DISABLED)

    def test_labels_are_stored_without_text(self):
        service, _, _ = make_service(self.tmp.name)
        service.observe("s1", turns(SECRET_TEXT, "no that's wrong, still broken"))
        answers = {"What kind of work": "refactor", "Which area": "Product engineering", "Scale": "1",
                   "previous work was wrong": "yes"}

        def respond(prompt):
            key = next(v for k, v in answers.items() if k in prompt)
            return jet_response(letter_for(prompt, key))

        FakeClient.responder = respond
        drain(service)
        turn_prompt = next(p for p in FakeClient.prompts if "previous work was wrong" in p)
        snapshot = service.snapshot()[service.session_key("s1")]
        self.assertEqual(snapshot["work_type"], "refactor")
        self.assertEqual(snapshot["area"], "Product engineering")
        self.assertEqual(snapshot["complexity"], "everyday")
        self.assertEqual(snapshot["corrections"], 1)
        self.assertIn("Assistant's previous message (end):\nDone: I changed the chart.", turn_prompt)
        with open(os.path.join(self.tmp.name, "work.sqlite3"), "rb") as handle:
            blob = handle.read()
        self.assertNotIn(b"zebra-kumquat", blob)
        self.assertNotIn(b"still broken", blob)
        self.assertNotIn(b"I changed the chart", blob)
        self.assertNotIn(b"s1", blob.replace(b"s1_", b""))
        self.assertEqual(service.observe("s1", turns(SECRET_TEXT, "no that's wrong, still broken")), 0)

    def test_model_missing_enters_setup_state_and_probes_later(self):
        service, _, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        FakeClient.digest_error = W.ClassifierError("setup", "model_missing")
        self.assertEqual(service.step(), W.SETUP_PROBE_S)
        self.assertEqual((service.state, service.reason), (W.STATE_SETUP, "model_missing"))
        self.assertLess(service.step(), W.SETUP_PROBE_S)
        FakeClient.digest_error = None
        clock.now += W.SETUP_PROBE_S + 1
        drain(service)
        self.assertEqual(len(service.queue), 0)

    def test_transport_failures_back_off_exponentially_and_keep_items(self):
        service, _, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        delays = []
        for _ in range(3):
            FakeClient.script = [W.ClassifierError("transport", "unreachable")]
            with mock.patch.object(W.random, "uniform", return_value=1.0):
                delays.append(service.step())
            clock.now += delays[-1] + 0.01
        self.assertEqual(delays, [5, 10, 20])
        self.assertEqual(service.state, W.STATE_BACKOFF)
        self.assertEqual(len(service.queue), 1)

    def test_item_failures_retry_then_become_terminal_unclear(self):
        service, _, clock = make_service(self.tmp.name, {"areas": list(W.DEFAULT_AREAS)[:2]})
        for attempt in range(W.MAX_ITEM_ATTEMPTS):
            service.observe("s1", [{"ts": clock.now, "text": "hello", "model": "m"}])
            FakeClient.script = [jet_response("Sure")] * 3
            service.pacer.tokens = 1.0
            service.pacer.last = -10
            service.step()
            clock.now += W.ITEM_RETRY_DELAYS_S[-1] + 1
        entry = service.snapshot()[service.session_key("s1")]
        self.assertEqual(entry["work_type"], "unclear")
        self.assertEqual(entry["area"], "Unclear")
        self.assertEqual(service.observe("s1", [{"ts": clock.now, "text": "hello", "model": "m"}]), 0)

    def test_manual_pause_unloads_and_stops_work(self):
        service, values, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        service.step()
        values["paused_until"] = "indefinite"
        service.observe("s2", turns("more work"))
        self.assertEqual(service.step(), 5.0)
        self.assertEqual(service.status()["state"], W.STATE_PAUSED)
        self.assertEqual(FakeClient.unloads, 1)
        calls = len(FakeClient.prompts)
        service.step()
        self.assertEqual(len(FakeClient.prompts), calls)
        values["paused_until"] = None
        drain(service)
        self.assertGreater(len(FakeClient.prompts), calls)

    def test_throttles_on_battery_and_load(self):
        power = {"value": "battery"}
        service, values, _ = make_service(self.tmp.name, power_probe=lambda: power["value"])
        service.observe("s1", turns("hello"))
        self.assertEqual(service.step(), W.THROTTLE_WAIT_S)
        self.assertEqual((service.state, service.reason), (W.STATE_THROTTLED, "on_battery"))
        values["pause_on_battery"] = False
        service.load_probe = lambda: 0.95
        service.retry_at = 0
        service.step()
        self.assertEqual(service.reason, "system_busy")

    def test_rate_limit_spaces_every_request(self):
        service, values, clock = make_service(self.tmp.name, {"rate_per_minute": 10})
        stamps = []
        FakeClient.responder = lambda prompt: stamps.append(clock.now) or jet_response("A")
        service.observe("s1", turns("a"))
        self.assertEqual(service.step(), 0.0)
        self.assertEqual(len(stamps), 5)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        self.assertTrue(all(gap >= 6.0 - 1e-6 for gap in gaps), gaps)

    def test_pause_mid_item_stops_before_the_next_request(self):
        service, values, clock = make_service(self.tmp.name)
        calls = []

        def respond(prompt):
            calls.append(prompt)
            values["paused_until"] = "indefinite"
            return jet_response("A")

        FakeClient.responder = respond
        service.observe("s1", turns("hello"))
        service.step()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(service.queue), 1)

    def test_clear_during_a_request_writes_nothing_afterwards(self):
        service, values, clock = make_service(self.tmp.name)
        FakeClient.responder = lambda prompt: service.clear() or jet_response("A")
        service.observe("s1", turns("hello"))
        service.step()
        self.assertEqual(service.snapshot(), {})
        self.assertEqual(service.status()["labels"], 0)

    def test_disable_clears_queued_text_immediately(self):
        service, values, _ = make_service(self.tmp.name)
        service.observe("s1", turns("hello", "more"))
        self.assertTrue(service.queue)
        values["enabled"] = False
        service.settings_changed()
        self.assertEqual(service.queue, [])

    def test_model_digest_change_is_detected(self):
        service, values, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        drain(service)
        version = service.labels_version
        with mock.patch.object(FakeClient, "model_digest", lambda self: "digest-2"):
            service.observe("s2", turns("next request"))
            service.step()
        self.assertEqual(service.model_digest, "digest-2")
        self.assertGreater(service.labels_version, version)

    def test_failed_items_schedule_a_content_free_retry(self):
        service, values, clock = make_service(self.tmp.name)
        FakeClient.responder = lambda prompt: jet_response("Sure")
        service.observe("s1", turns("hello"))
        service.step()
        key = service.session_key("s1")
        self.assertEqual(service.ledger.next_backlog(5, clock.now), [])
        self.assertEqual(service.ledger.next_backlog(5, clock.now + W.ITEM_RETRY_DELAYS_S[0] + 1), [key])

    def test_overflow_is_credited_to_the_session_it_came_from(self):
        service, values, clock = make_service(self.tmp.name)
        with mock.patch.object(W, "QUEUE_LIMIT", 2):
            service.observe("old", [{"ts": clock.now - 5000 + i, "text": f"t{i}", "model": "m"} for i in range(2)])
            service.observe("live", [{"ts": clock.now - 10 + i, "text": f"l{i}", "model": "m"} for i in range(2)])
        with service.ledger._connect() as connection:
            rows = dict(connection.execute("SELECT session_key, pending FROM work_backlog").fetchall())
        self.assertEqual(rows, {service.session_key("old"): 2})

    def test_backlog_rows_that_yield_nothing_are_dropped_after_refill(self):
        service, values, clock = make_service(self.tmp.name, refill=lambda keys: None)
        service.ledger.upsert_backlog(service.session_key("gone"), clock.now, 3, clock.now)
        service.step()
        self.assertEqual(service.ledger.backlog_pending(), 0)

    def test_pacing_wait_does_not_spin_when_wake_is_set(self):
        clock = Clock()
        calls = []
        service = None

        def event_sleep(seconds):
            # Mirror Event.wait: a set event returns immediately without time passing.
            calls.append(seconds)
            if service.wake.is_set():
                return
            clock.now += seconds

        FakeClient.script, FakeClient.prompts, FakeClient.responder, FakeClient.digest_error = [], [], None, None
        values = W.normalize_settings({"enabled": True, "backfill_days": 0, "rate_per_minute": 10})
        service = W.WorkInsightsService(os.path.join(self.tmp.name, "spin.sqlite3"), lambda: values,
                                        client_factory=FakeClient, clock=clock, monotonic=clock,
                                        sleep=event_sleep, load_probe=lambda: 0.1)
        service.observe("s1", turns("hello"))
        service.wake.set()
        service.step()
        self.assertEqual(len(FakeClient.prompts), 5)
        self.assertLess(len(calls), 12)

    def test_multiple_failures_in_one_session_all_keep_a_retry(self):
        service, values, clock = make_service(self.tmp.name)
        FakeClient.responder = lambda prompt: jet_response("Sure")
        session = [{"ts": clock.now - 100 + i, "text": f"turn {i}", "model": "m", "context": "done"} for i in range(3)]
        service.observe("s1", session)
        for _ in range(3):
            service.step()
            clock.now += 30
        key = service.session_key("s1")
        service.observe("s1", session)
        self.assertEqual(service.ledger.next_backlog(5, clock.now + 3_600), [key])
        clock.now += W.ITEM_RETRY_DELAYS_S[0] + 1
        FakeClient.responder = lambda prompt: jet_response(letter_for(prompt, "debug")) \
            if "What kind of work" in prompt else jet_response("A") if "Which area" in prompt \
            else jet_response("1") if "Scale" in prompt else jet_response("no")
        drain(service)
        entry = service.snapshot()[key]
        self.assertEqual(entry.get("correction_labels"), 2)

    def test_digest_is_rechecked_when_the_model_changes(self):
        service, values, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        drain(service)
        checks = []
        with mock.patch.object(FakeClient, "model_digest", lambda self: checks.append(self.model) or "d"):
            values["model"] = "other-model"
            service.observe("s2", turns("more"))
            service.step()
        self.assertEqual(set(checks), {"other-model"})

    def test_turn_keys_ignore_text(self):
        service, _, _ = make_service(self.tmp.name)
        self.assertEqual(service._turn_key("s1", 2), service._turn_key("s1", 2))
        self.assertNotEqual(service._turn_key("s1", 2), service._turn_key("s1", 3))

    def test_v1_ledger_is_recreated_not_dropped_in_place(self):
        import sqlite3
        path = os.path.join(self.tmp.name, "old.sqlite3")
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE work_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO work_metadata VALUES ('salt', 'old-salt-value')")
            connection.execute("PRAGMA user_version = 1")
        ledger = W.LabelLedger(path)
        self.assertNotEqual(ledger.salt, "old-salt-value")
        with open(path, "rb") as handle:
            self.assertNotIn(b"old-salt-value", handle.read())

    def test_locked_ledger_is_never_deleted(self):
        import sqlite3
        path = os.path.join(self.tmp.name, "locked.sqlite3")
        ledger = W.LabelLedger(path)
        salt = ledger.salt
        holder = sqlite3.connect(path, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            with mock.patch.object(W.LabelLedger, "_connect",
                                   lambda self: sqlite3.connect(self.path, timeout=0.05)):
                with self.assertRaises(sqlite3.OperationalError):
                    W.LabelLedger(path)
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        self.assertEqual(W.LabelLedger(path).salt, salt)

    def test_observe_after_clear_with_a_stale_generation_queues_nothing(self):
        service, values, _ = make_service(self.tmp.name)
        original = service._turn_key

        def racing_key(row_id, ordinal):
            key = original(row_id, ordinal)
            if service.generation == 0:
                service.clear()
            return key

        service._turn_key = racing_key
        self.assertEqual(service.observe("s1", turns("hello")), 0)
        self.assertEqual(service.queue, [])

    def test_clear_before_intake_reads_generation_discards_old_salt_keys(self):
        service, values, _ = make_service(self.tmp.name)
        original = service.session_key
        state = {"cleared": False}

        def racing_session_key(row_id):
            key = original(row_id)
            if not state["cleared"]:
                state["cleared"] = True
                service.clear()
            return key

        service.session_key = racing_session_key
        self.assertEqual(service.observe("s1", turns("hello")), 0)
        self.assertEqual(service.queue, [])

    def test_setup_state_persists_with_an_empty_queue(self):
        service, values, clock = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        FakeClient.digest_error = W.ClassifierError("setup", "model_missing")
        service.step()
        service.queue.clear()
        service.queued.clear()
        clock.now += W.SETUP_PROBE_S + 1
        service.step()
        self.assertEqual((service.state, service.reason), (W.STATE_SETUP, "model_missing"))

    def test_model_repointed_to_cloud_between_requests_sends_nothing_more(self):
        service, values, clock = make_service(self.tmp.name)
        calls = []

        def respond(prompt):
            calls.append(prompt)
            FakeClient.digest_error = W.ClassifierError("setup", "remote_model")
            return jet_response("A")

        FakeClient.responder = respond
        service.observe("s1", turns("hello"))
        service.step()
        self.assertEqual(len(calls), 1)
        self.assertEqual((service.state, service.reason), (W.STATE_SETUP, "remote_model"))

    def test_an_item_with_several_failed_questions_adds_one_retry(self):
        service, values, clock = make_service(self.tmp.name)
        FakeClient.responder = lambda prompt: jet_response("Sure")
        service.observe("s1", turns("hello"))
        service.step()
        self.assertEqual(service.ledger.backlog_pending(), 1)

    def test_failed_clear_reopens_the_ledger_and_requeues(self):
        recovered = []
        service, values, clock = make_service(self.tmp.name, recovered=lambda: recovered.append(1))
        with mock.patch.object(W.LabelLedger, "clear", side_effect=OSError("disk")):
            service.clear()
        self.assertIsNone(service.ledger)
        clock.now += W.STORAGE_RETRY_S + 1
        service.state = W.STATE_IDLE
        service.step()
        self.assertIsNotNone(service.ledger)
        self.assertEqual(recovered, [1])

    def test_latency_baseline_adapts(self):
        service, _, _ = make_service(self.tmp.name)
        for _ in range(W.LATENCY_WINDOW):
            service._record_latency(1.0)
        self.assertEqual(service.baseline, 1.0)
        for _ in range(200):
            service._record_latency(2.0)
        self.assertGreater(service.baseline, 1.9)

    def test_queue_overflow_goes_to_content_free_backlog_and_refills(self):
        refilled = []
        service, _, clock = make_service(self.tmp.name, refill=lambda keys: refilled.extend(keys) or keys)
        with mock.patch.object(W, "QUEUE_LIMIT", 2), mock.patch.object(W, "QUEUE_LOW_WATER", 1):
            clock.now += 10_000
            service.observe("big", turns("one", "two", "three", "four"))
            self.assertEqual(len(service.queue), 2)
            self.assertEqual(service.ledger.backlog_pending(), 2)
            service.queue.clear()
            service.queued.clear()
            service.step()
        self.assertIn(service.session_key("big"), refilled)

    def test_area_edit_requeues_only_area(self):
        service, values, _ = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        drain(service)
        values["areas"] = [{"name": "Client delivery", "description": "customer work"},
                           {"name": "Personal", "description": "personal"}]
        service.settings_changed()
        self.assertEqual(service.snapshot()[service.session_key("s1")].get("area"), None)
        service.observe("s1", turns("hello"))
        self.assertEqual(service.queue[0].questions, ("area",))

    def test_backfill_horizon_skips_old_sessions(self):
        service, _, clock = make_service(self.tmp.name, {"backfill_days": 30})
        old = [{"ts": clock.now - 40 * 86400, "text": "old", "model": "m"}]
        self.assertEqual(service.observe("old", old), 0)

    def test_worker_exceptions_are_supervised(self):
        service, _, _ = make_service(self.tmp.name)
        stop = threading.Event()
        calls = []

        def boom():
            calls.append(1)
            if len(calls) > 2:
                stop.set()
            raise RuntimeError("secret text inside")

        service.step = boom
        service.run_forever(stop)
        self.assertEqual((service.state, service.reason), (W.STATE_BACKOFF, "internal_error"))

    def test_status_eta_uses_measured_item_rate(self):
        service, values, clock = make_service(self.tmp.name)
        self.assertEqual(service.status()["state"], W.STATE_IDLE)
        for index in range(10):
            service._rate_window.append(clock.now - 60 + index * 6)
        service.queue.extend([None] * 30)
        self.assertAlmostEqual(service.status()["eta_s"], int(30 / (9 / 0.9) * 60), delta=2)
        service.queue.clear()

    def test_clear_removes_labels_and_rotates_salt(self):
        service, _, _ = make_service(self.tmp.name)
        service.observe("s1", turns("hello"))
        drain(service)
        salt = service.ledger.salt
        service.clear()
        self.assertEqual(service.snapshot(), {})
        self.assertNotEqual(service.ledger.salt, salt)


class FakeOllama(http.server.BaseHTTPRequestHandler):
    responses = {}

    def log_message(self, *args):
        pass

    def _reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        status, body = FakeOllama.responses.get(self.path, (404, {"error": "not found"}))
        self._reply(status, body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        status, body = FakeOllama.responses.get(self.path, (404, {"error": "not found"}))
        self._reply(status, body)


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_classify_and_digest(self):
        FakeOllama.responses = {
            "/api/tags": (200, {"models": [{"name": "token-meter-jet:latest", "digest": "abc123"}]}),
            "/api/chat": (200, jet_response("B")),
        }
        client = W.OllamaClient(self.url, "token-meter-jet")
        self.assertEqual(client.model_digest(), "abc123")
        self.assertEqual(client.classify("p", 5)["message"]["content"], "B")

    def test_missing_model_and_server_errors_are_classified(self):
        FakeOllama.responses = {"/api/tags": (200, {"models": []}),
                                "/api/chat": (500, {"error": "boom"})}
        client = W.OllamaClient(self.url, "token-meter-jet")
        with self.assertRaises(W.ClassifierError) as missing:
            client.model_digest()
        self.assertEqual(missing.exception.kind, "setup")
        with self.assertRaises(W.ClassifierError) as failed:
            client.classify("p", 5)
        self.assertEqual(failed.exception.kind, "transport")

    def test_remote_and_cloud_models_are_refused(self):
        for entry in ({"name": "token-meter-jet:latest", "remote_host": "https://ollama.com"},
                      {"name": "token-meter-jet:latest", "remote_model": "gpt-oss:120b"},
                      {"name": "token-meter-jet-cloud"},
                      {"name": "token-meter-jet:cloud"}):
            FakeOllama.responses = {"/api/tags": (200, {"models": [dict(entry, digest="x")]})}
            with self.assertRaises(W.ClassifierError) as caught:
                name = entry["name"].split(":")[0] if entry["name"].endswith(":latest") else entry["name"]
                W.OllamaClient(self.url, name).model_digest()
            self.assertEqual(caught.exception.reason, "remote_model")

    def test_unreachable_is_transport(self):
        self.server.shutdown()
        self.server.server_close()
        self.addCleanup(lambda: None)
        with self.assertRaises(W.ClassifierError) as caught:
            W.OllamaClient(self.url, "m").version()
        self.assertEqual((caught.exception.kind, caught.exception.reason), ("transport", "unreachable"))


def row(row_id, project="alpha", runtime="Codex", cost=2.0, day="2026-09-10", turns_=3, model="gpt-5.6"):
    return {"id": row_id, "project": project, "runtime": runtime, "provider": "codex", "cost": cost,
            "start": f"{day} 10:00", "_day_cost": {day: cost},
            "model_stats": [{"model": model, "cost": cost, "tokens": 100}],
            "_language_signal_events": {"positive": [{"day": day}] * turns_}}


class DomainTests(unittest.TestCase):
    AREAS = [{"name": "Product engineering", "description": "d"}, {"name": "Personal", "description": "d"}]

    def build(self, rows, labels, **kwargs):
        prices = {"gpt-5.6": 10.0, "cheap": 1.0, "mid": 4.0}
        return domain.build_work_insights(rows, labels, lambda rid: rid, self.AREAS,
                                          lambda m, p: prices.get(m), today="2026-09-30", **kwargs)

    def test_allocation_measures_and_pending(self):
        rows = [row("a"), row("b", cost=6.0, turns_=1), row("c", day="2026-08-02")]
        labels = {"a": {"area": "Product engineering", "work_type": "debug", "corrections": 1, "correction_labels": 2},
                  "b": {"area": "Unclear"}}
        out = self.build(rows, labels)
        september = next(b for b in out["allocation"] if b["month"] == "2026-09")
        self.assertEqual(september["turns"], {"Product engineering": 3, "Unclear": 1})
        self.assertEqual(september["spend"]["Unclear"], 6.0)
        self.assertTrue(september["partial"])
        august = next(b for b in out["allocation"] if b["month"] == "2026-08")
        self.assertEqual(august["sessions"], {"Pending": 1})
        self.assertEqual(out["areas"][-2:], ["Unclear", "Pending"])

    def test_child_rows_are_excluded_but_parents_with_children_are_kept(self):
        child = row("kid")
        child["_agent_records"] = [{"parent_id": "p1"}]
        parent = row("parent")
        parent["_agent_records"] = [{"id": "root", "parent_id": None}, {"id": "c", "parent_id": "root"}]
        self.assertTrue(domain.is_child_row(child))
        self.assertFalse(domain.is_child_row(parent))
        out = self.build([child, parent], {})
        self.assertEqual(out["coverage"]["sessions"], 1)

    def test_workstream_rework_counts_once_in_the_start_month(self):
        spanning = row("a")
        spanning["_language_signal_events"] = {"positive": [{"day": "2026-08-30"}, {"day": "2026-09-02"}]}
        spanning["start"] = "2026-08-30 10:00"
        spanning["_day_cost"] = {"2026-08-30": 1.0, "2026-09-02": 1.0}
        labels = {"a": {"area": "Personal", "corrections": 1, "correction_labels": 1}}
        out = self.build([spanning], labels)
        august = out["workstreams"]["2026-08"]["rows"][0]["rework"]
        september = out["workstreams"]["2026-09"]["rows"][0]["rework"]
        self.assertEqual(august["samples"], 1)
        self.assertIsNone(september)

    def test_workstreams_economics_and_few_samples(self):
        rows = [row("a"), row("b", project="beta", turns_=5)]
        labels = {"a": {"area": "Personal", "work_type": "docs", "corrections": 1, "correction_labels": 2},
                  "b": {"area": "Personal", "work_type": "docs", "corrections": 0, "correction_labels": 4}}
        out = self.build(rows, labels)
        ws = out["workstreams"]["2026-09"]["rows"]
        self.assertEqual([w["project"] for w in ws], ["beta", "alpha"])
        self.assertAlmostEqual(sum(w["share"] for w in ws), 1.0)
        docs = out["economics"][0]
        self.assertEqual((docs["work_type"], docs["sessions"], docs["cost_per_session"]), ("docs", 2, 2.0))
        self.assertTrue(docs["rework"]["few_samples"])
        self.assertAlmostEqual(docs["rework"]["rate"], 1 / 6)

    def test_right_sizing_flags_premium_routine(self):
        rows = [row("a", model="gpt-5.6"), row("b", model="cheap"), row("c", model="mid")]
        labels = {"a": {"complexity": "routine"}, "b": {"complexity": "complex"}, "c": {"complexity": "everyday"}}
        cells = {(c["complexity"], c["tier"]): c for c in self.build(rows, labels)["right_sizing"]["cells"]}
        self.assertEqual(cells[("routine", "premium")]["flag"], "possible_overspend")
        self.assertEqual(cells[("complex", "light")]["sessions"], 1)

    def test_filters_and_period(self):
        rows = [row("a"), row("b", runtime="Claude Code", project="beta", day="2025-01-05")]
        out = self.build(rows, {}, months=3, runtime="Codex")
        self.assertEqual(out["months"], ["2026-09"])
        self.assertEqual(out["filters"]["runtimes"], ["Claude Code", "Codex"])
        self.assertEqual(set(out["filters"]["projects"]), {"alpha", "beta"})


class AppContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = os.path.join(self.tmp.name, "settings.json")

    def test_settings_round_trip_validation_and_pause(self):
        result = meter.set_work_insights_settings({"enabled": True, "rate_per_minute": 40}, self.settings)
        self.assertTrue(result["ok"])
        self.assertTrue(result["requeue"])
        loaded = meter.work_insights_settings(self.settings)
        self.assertEqual((loaded["enabled"], loaded["rate_per_minute"]), (True, 40))
        self.assertFalse(meter.set_work_insights_settings({"ollama_url": "http://10.1.1.1"}, self.settings)["ok"])
        self.assertFalse(meter.set_work_insights_settings({"unknown": 1}, self.settings)["ok"])
        self.assertFalse(meter.set_work_insights_settings({"pause": "forever"}, self.settings)["ok"])
        self.assertTrue(meter.set_work_insights_settings({"pause": "1h"}, self.settings)["ok"])
        self.assertIsInstance(meter.work_insights_settings(self.settings)["paused_until"], float)
        again = meter.set_work_insights_settings({"pause": "resume"}, self.settings)
        self.assertIsNone(again["work_insights"]["paused_until"])
        with open(self.settings, encoding="utf-8") as handle:
            self.assertIn("work_insights", json.load(handle))

    def test_pause_until_tomorrow_is_six_am(self):
        import datetime
        now = datetime.datetime(2026, 9, 30, 22, 15).timestamp()
        until = datetime.datetime.fromtimestamp(meter._pause_until("tomorrow", now))
        self.assertEqual((until.day, until.hour, until.minute), (1, 6, 0))

    def test_work_payload_is_bounded_and_text_free(self):
        rows = (row("a"),)
        with mock.patch.object(meter, "TOKEN_METER_SETTINGS", self.settings), \
                mock.patch.dict(meter._xsess, {"data": {"ok": True}, "internal_rows": rows}):
            payload, status = meter.work_insights_state("6")
            self.assertEqual(status, 200)
            self.assertEqual(meter.work_insights_state("7")[1], 400)
            self.assertEqual(meter.work_insights_state("6", project="nope")[1], 404)
        encoded = json.dumps(payload)
        self.assertNotIn("_day_cost", encoded)
        self.assertNotIn("_language_signal_events", encoded)
        self.assertEqual(set(payload), {"ok", "settings", "status", "insights"})

    def test_routes_are_registered(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "token_meter", "app.py"), encoding="utf-8") as handle:
            source = handle.read()
        for route in ('"/work"', '"/settings/work-insights"', '"/work-insights/pause"', '"/work-insights/clear"'):
            self.assertTrue(route in source, route)


class SurfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "page.html"), encoding="utf-8") as handle:
            cls.page = handle.read()
        with open(os.path.join(root, "menubar", "TokenMeterMenuBar.swift"), encoding="utf-8") as handle:
            cls.swift = handle.read()
        cls.root = root

    def test_work_tab_sits_between_efficiency_and_git_without_a_digit_shortcut(self):
        rail = [self.page.index(f"id=tab-{name}") for name in ("efficiency", "work", "git")]
        self.assertEqual(rail, sorted(rail))
        button = self.page[self.page.index("id=tab-work"):].split("</button>", 1)[0]
        self.assertNotIn("aria-keyshortcuts", button)
        self.assertIn("{id:'work',label:'Work'", self.page)
        self.assertIn("if(h==='work'){", self.page)

    def test_work_page_shows_estimates_unclear_and_pending(self):
        for marker in ("id=view-work", "Monthly activity allocation", "Workstreams", "Cost by work type",
                       "Model right-sizing", "Possible overspend (estimate)", "View as table",
                       "text goes only to Ollama on this machine", "'var(--w-pending)'", "'var(--w-unclear)'"):
            self.assertTrue(marker in self.page, marker)

    def test_settings_card_explains_what_text_is_read(self):
        self.assertIn("id=work-insights-settings", self.page)
        self.assertIn("reads the prompts you typed, plus the last few lines of the assistant reply", self.page)
        self.assertIn("cloud models are refused", self.page)
        self.assertIn("Token Meter stores labels, never text", self.page)
        self.assertIn("./scripts/setup-work-classifier", self.page)

    def test_menu_bar_offers_pause_and_resume(self):
        for marker in ('"Pause work insights"', '"Resume work insights"', '/work-insights/pause"',
                       'dict["work_insights"]', '("Until tomorrow", "tomorrow")'):
            self.assertTrue(marker in self.swift, marker)

    def test_setup_script_is_executable_and_uses_int4_import(self):
        path = os.path.join(self.root, "scripts", "setup-work-classifier")
        self.assertTrue(os.access(path, os.X_OK))
        with open(path, encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn('ollama create "$MODEL_NAME" -q int4', script)
        self.assertIn("file_digest", script)
        self.assertIn('COMMIT="fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a"', script)
        self.assertNotIn("sudo", script)


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = os.path.join(self.tmp.name, "settings.json")
        self.db = os.path.join(self.tmp.name, "work.sqlite3")

    def test_disabled_feature_creates_no_ledger(self):
        with mock.patch.object(meter, "TOKEN_METER_SETTINGS", self.settings), \
                mock.patch.object(meter, "TOKEN_METER_WORK_INSIGHTS_DB", self.db), \
                mock.patch.object(meter, "_work_service_instance", None):
            self.assertEqual(meter.work_insights_status()["state"], "disabled")
            meter.clear_work_insights()
            meter.notify_work_insights()
        self.assertFalse(os.path.exists(self.db))

    def test_thread_local_turns_are_cleared_when_a_summarizer_raises(self):
        class Boom:
            def summarize_legacy(self, source, conn=None):
                meter.analyze_language_signal_turns([{"ts": 1, "text": SECRET_TEXT, "model": "m"}])
                raise RuntimeError("parse failure")

        registry = mock.Mock()
        registry.get.return_value = Boom()
        source = {"path": os.path.join(self.tmp.name, "x.jsonl"), "provider": "codex", "id": "x"}
        with mock.patch.object(meter, "work_insights_settings", return_value=W.normalize_settings({"enabled": True})), \
                mock.patch.object(meter, "runtime_registry", return_value=registry), \
                mock.patch.object(meter, "source_revision_signature", return_value="sig"):
            with self.assertRaises(RuntimeError):
                meter.session_summary(source)
        self.assertIsNone(getattr(meter._WORK_TURNS, "turns", None))

    def post(self, path, body, token=True):
        server = meter.TokenMeterHTTPServer(("127.0.0.1", 0), meter.H)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            data = json.dumps(body)
            headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
            if token:
                headers["X-Token-Meter-Action"] = meter._ACTION_TOKEN
            conn.request("POST", path, body=data, headers=headers)
            response = conn.getresponse()
            payload = json.loads(response.read())
            conn.close()
            return response.status, payload
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_post_routes_validate_over_http(self):
        with mock.patch.object(meter, "TOKEN_METER_SETTINGS", self.settings), \
                mock.patch.object(meter, "TOKEN_METER_WORK_INSIGHTS_DB", self.db), \
                mock.patch.object(meter, "_work_service_instance", None):
            self.assertEqual(self.post("/settings/work-insights", {"ollama_url": "http://10.0.0.2"})[0], 400)
            self.assertEqual(self.post("/settings/work-insights", {"bogus": 1})[0], 400)
            self.assertEqual(self.post("/work-insights/pause", {"duration": "forever"})[0], 400)
            self.assertEqual(self.post("/work-insights/clear", {"confirm": False})[0], 400)
            self.assertEqual(self.post("/work-insights/pause", {"duration": "1h"}, token=False)[0], 403)
            status, payload = self.post("/work-insights/pause", {"duration": "1h"})
            self.assertEqual(status, 200)
            self.assertIsInstance(payload["work_insights"]["paused_until"], float)
        self.assertFalse(os.path.exists(self.db))


class SessionTagProjectionTests(unittest.TestCase):
    def test_tags_are_enums_and_counts_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        service, values, _ = make_service(tmp.name)
        service.observe("sess-1", turns(SECRET_TEXT, "that's wrong"))
        drain(service)
        with mock.patch.object(meter, "work_insights_service", return_value=service), \
                mock.patch.object(meter, "work_insights_settings", return_value=values):
            tags = meter.work_session_tags("sess-1")
            payload = meter.dashboard_state_payload({"source": {"id": "sess-1"}})
        self.assertEqual(set(tags), {"area", "work_type", "complexity", "corrections", "labeled_turns"})
        self.assertEqual(payload["work_tags"], tags)
        self.assertNotIn("zebra", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
