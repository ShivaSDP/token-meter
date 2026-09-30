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
    service = W.WorkInsightsService(
        os.path.join(tmp, "work.sqlite3"), lambda: values, client_factory=FakeClient,
        clock=clock, monotonic=clock, sleep=lambda s: None,
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
        self.assertEqual(W.validate_ollama_url("http://localhost"), "http://localhost:11434")
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

    def test_rate_limit_spaces_requests(self):
        service, values, clock = make_service(self.tmp.name, {"rate_per_minute": 10})
        service.observe("s1", turns("a"))
        service.observe("s2", turns("b"))
        self.assertEqual(service.step(), 0.0)
        wait = service.step()
        self.assertGreater(wait, 0)
        self.assertEqual(len(FakeClient.prompts), 5)

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

    def test_child_rows_are_excluded(self):
        child = row("kid")
        child["_agent_records"] = [{"parent_id": "p1"}]
        out = self.build([child], {})
        self.assertEqual(out["coverage"]["sessions"], 0)

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
                       "no prompt text leaves this machine", "'var(--w-pending)'", "'var(--w-unclear)'"):
            self.assertTrue(marker in self.page, marker)

    def test_settings_card_explains_what_text_is_read(self):
        self.assertIn("id=work-insights-settings", self.page)
        self.assertIn("reads only the prompts you typed", self.page)
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
        self.assertNotIn("sudo", script)


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
