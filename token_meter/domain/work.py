"""Runtime-neutral work insights: allocation, workstreams, work-type economics, rework, right-sizing.

Pure functions over cached session summary rows and content-free classifier labels.
"""

import collections
import datetime
import statistics

UNCLEAR = "Unclear"
PENDING = "Pending"
MIN_RATE_SAMPLES = 20
MAX_MONTHS = 24
MAX_WORKSTREAMS = 30
COMPLEXITY_GROUPS = (
    ("routine", ("routine",)),
    ("everyday", ("everyday",)),
    ("complex", ("complex", "high_impact")),
)
TIERS = ("light", "standard", "premium")
WORK_TYPE_ORDER = ("debug", "feature", "refactor", "docs", "explore", "review", "ops", "other", "unclear")


def is_child_row(row):
    """A row is a child run only when none of its agent records is a root (a parent keeps its children)."""
    records = [r for r in row.get("_agent_records") or () if isinstance(r, dict)]
    return bool(records) and all(r.get("parent_id") for r in records)


def turn_days(row):
    events = (row.get("_language_signal_events") or {}).get("positive") or []
    return [str(event.get("day") or "") for event in events]


def primary_model(row):
    stats = [s for s in row.get("model_stats") or [] if isinstance(s, dict) and s.get("model")]
    if stats:
        return max(stats, key=lambda s: (float(s.get("cost") or 0), int(s.get("tokens") or 0)))["model"]
    models = row.get("models") or []
    return models[0] if models else ""


def price_tiers(rows, output_price, with_prices=False):
    """Map (runtime, model) to light/standard/premium by terciles of catalog output price."""
    prices = {}
    for row in rows:
        model = primary_model(row)
        if not model:
            continue
        key = (row.get("runtime") or "", model)
        if key not in prices:
            price = output_price(model, row.get("provider") or "")
            if price and price > 0:
                prices[key] = float(price)
    distinct = sorted(set(prices.values()))
    if not distinct:
        return ({}, {}) if with_prices else {}
    if len(distinct) == 1:
        tiers = {key: "standard" for key in prices}
    else:
        def tier(price):
            position = distinct.index(price) / (len(distinct) - 1)
            return "light" if position < 1 / 3 else "premium" if position > 2 / 3 else "standard"

        tiers = {key: tier(price) for key, price in prices.items()}
    if not with_prices:
        return tiers
    by_tier = collections.defaultdict(list)
    for key, name in tiers.items():
        by_tier[name].append(prices[key])
    return tiers, {name: statistics.median(values) for name, values in by_tier.items()}


def _month(day):
    return day[:7] if len(day) >= 7 else ""


def _rate(corrections, samples):
    if samples <= 0:
        return None
    return {"rate": corrections / samples, "samples": samples, "few_samples": samples < MIN_RATE_SAMPLES}


OUTCOMES = ("single_shot", "accepted", "recovered", "ended_on_pushback", "unclear", "pending")
POSITION_BUCKETS = ((1, 2, "1–2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 20, "11–20"), (21, 10**9, "21+"))
MAX_FIT_MODELS = 5
MAX_HEADLINES = 4
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max", "ultra")
HIGH_EFFORTS = ("xhigh", "max", "ultra")


def session_outcome(turns, sequence, pending=False):
    """Classify a session from its ordered follow-up pushback labels.

    A session is classified only once it has no queued or backlogged work (``pending``);
    sessions whose labels are all low-confidence are Unclear rather than counted as accepted.
    """
    if turns <= 1:
        return "single_shot"
    if pending or not sequence:
        return "pending"
    confident = [value for _ordinal, value in sequence if value is not None]
    if not confident:
        return "unclear"
    if not any(confident):
        return "accepted"
    return "ended_on_pushback" if confident[-1] else "recovered"


def rework_share(turns, sequence):
    """Share of a session's turns that came after its first pushback (0 when none)."""
    first = next((ordinal for ordinal, value in sequence if value), None)
    if first is None or turns <= 0:
        return 0.0
    return max(0.0, min(1.0, (turns - first) / turns))


def build_work_insights(rows, labels, key_for, areas, output_price, months=6,
                        runtime="", project="", today="", corrections_for=None, pending_keys=None):
    """Aggregate labeled sessions. ``months`` 0 means all history."""
    area_names = [a["name"] for a in areas]
    tiers, tier_prices = price_tiers(rows, output_price, with_prices=True)
    sessions, runtime_options, project_options = _prepare_sessions(
        rows, labels, key_for, area_names, tiers, runtime, project, pending_keys)
    all_months = _window_months(sessions, months)
    month_set = set(all_months)
    segments = area_names + [UNCLEAR, PENDING]
    return _aggregate(sessions, all_months, month_set, segments, area_names, tiers, tier_prices,
                      runtime_options, project_options, today, corrections_for)


def _prepare_sessions(rows, labels, key_for, area_names, tiers, runtime, project, pending_keys):
    sessions = []
    runtime_options, project_options = set(), collections.Counter()
    for row in rows:
        if is_child_row(row):
            continue
        if row.get("runtime"):
            runtime_options.add(row["runtime"])
        if row.get("project"):
            project_options[row["project"]] += 1
        if runtime and (row.get("runtime") or "") != runtime:
            continue
        if project and (row.get("project") or "") != project:
            continue
        entry = labels.get(key_for(row.get("id") or "")) or {}
        area = entry.get("area") or PENDING
        if area not in area_names and area not in (UNCLEAR, PENDING):
            area = PENDING
        days = [d for d in turn_days(row) if d]
        start_day = (row.get("start") or "")[:10]
        sessions.append({
            "row": row, "entry": entry, "area": area,
            "work_type": entry.get("work_type") or "",
            "complexity": entry.get("complexity") or "",
            "days": days, "start_month": _month(start_day or (days[0] if days else "")),
            "turns": len(turn_days(row)),
            "corrections": int(entry.get("corrections") or 0),
            "correction_labels": int(entry.get("correction_labels") or 0),
            "tier": tiers.get((row.get("runtime") or "", primary_model(row))),
            "model": primary_model(row),
            "effort": str(row.get("reasoning_effort") or "").lower(),
            "sequence": [],
            "pending": bool(pending_keys) and key_for(row.get("id") or "") in pending_keys,
        })
    return sessions, runtime_options, project_options


def _session_months(s):
    return {m for m in [s["start_month"], *map(_month, s["days"]),
                        *map(_month, (s["row"].get("_day_cost") or {}))] if m}


def _window_months(sessions, months):
    all_months = sorted(set().union(*(_session_months(s) for s in sessions))) if sessions else []
    if months:
        all_months = all_months[-months:]
    return all_months[-MAX_MONTHS:]


def _attach_sequences(sessions, corrections_for):
    if corrections_for is None:
        return
    for s in sessions:
        if s["turns"] > 1 and s["correction_labels"]:
            s["sequence"] = corrections_for(s["row"].get("id") or "", s["turns"])


def _aggregate(sessions, all_months, month_set, segments, area_names, tiers, tier_prices,
               runtime_options, project_options, today, corrections_for):
    allocation = []
    for month in all_months:
        bucket = {"month": month, "partial": bool(today and today[:7] == month),
                  "turns": dict.fromkeys(segments, 0), "sessions": dict.fromkeys(segments, 0),
                  "spend": dict.fromkeys(segments, 0.0)}
        allocation.append(bucket)
    by_month = {b["month"]: b for b in allocation}
    for s in sessions:
        for day in s["days"]:
            bucket = by_month.get(_month(day))
            if bucket:
                bucket["turns"][s["area"]] += 1
        if s["start_month"] in by_month:
            by_month[s["start_month"]]["sessions"][s["area"]] += 1
        for day, cost in (s["row"].get("_day_cost") or {}).items():
            bucket = by_month.get(_month(str(day)))
            if bucket:
                bucket["spend"][s["area"]] += float(cost or 0)
    for bucket in allocation:
        for measure in ("turns", "sessions", "spend"):
            values = bucket[measure]
            bucket[measure] = {k: (round(v, 6) if measure == "spend" else v) for k, v in values.items() if v}
            bucket[measure + "_total"] = round(sum(values.values()), 6) if measure == "spend" else sum(values.values())

    workstreams = {}
    for month in all_months:
        groups = {}
        for s in sessions:
            month_turns = sum(1 for d in s["days"] if _month(d) == month)
            month_spend = sum(float(c or 0) for d, c in (s["row"].get("_day_cost") or {}).items()
                              if _month(str(d)) == month)
            if not month_turns and not month_spend:
                continue
            key = (s["row"].get("project") or "", s["area"])
            group = groups.setdefault(key, {"project": key[0], "area": key[1], "turns": 0, "sessions": 0,
                                            "spend": 0.0, "corrections": 0, "correction_labels": 0,
                                            "work_types": collections.Counter()})
            group["turns"] += month_turns
            group["sessions"] += 1
            group["spend"] += month_spend
            if s["start_month"] == month:
                # Corrections are session totals; count them once, in the month the session started.
                group["corrections"] += s["corrections"]
                group["correction_labels"] += s["correction_labels"]
            if s["work_type"]:
                group["work_types"][s["work_type"]] += 1
        total_turns = sum(g["turns"] for g in groups.values()) or 0
        rows_out = []
        for group in sorted(groups.values(), key=lambda g: (-g["turns"], -g["spend"])):
            dominant = group["work_types"].most_common(1)
            rows_out.append({
                "project": group["project"], "area": group["area"], "turns": group["turns"],
                "share": group["turns"] / total_turns if total_turns else 0.0,
                "sessions": group["sessions"], "spend": round(group["spend"], 6),
                "rework": _rate(group["corrections"], group["correction_labels"]),
                "work_type": dominant[0][0] if dominant else "",
            })
        workstreams[month] = {"total_turns": total_turns, "rows": rows_out[:MAX_WORKSTREAMS],
                              "truncated": len(rows_out) > MAX_WORKSTREAMS}

    in_window = [s for s in sessions if s["start_month"] in month_set]
    _attach_sequences(in_window, corrections_for)
    economics = []
    by_type = collections.defaultdict(list)
    for s in in_window:
        if s["work_type"]:
            by_type[s["work_type"]].append(s)
    for work_type in WORK_TYPE_ORDER:
        group = by_type.get(work_type)
        if not group:
            continue
        spend = sum(float(s["row"].get("cost") or 0) for s in group)
        economics.append({
            "work_type": work_type, "sessions": len(group), "spend": round(spend, 6),
            "cost_per_session": spend / len(group),
            "median_turns": statistics.median([s["turns"] for s in group]),
            "rework": _rate(sum(s["corrections"] for s in group), sum(s["correction_labels"] for s in group)),
        })

    weekly = collections.defaultdict(lambda: [0, 0])
    models = collections.defaultdict(lambda: [0, 0, 0.0])
    for s in in_window:
        if s["correction_labels"]:
            week = _week_start(s["days"][0]) if s["days"] else ""
            if week:
                weekly[week][0] += s["corrections"]
                weekly[week][1] += s["correction_labels"]
            key = (s["model"], s["row"].get("runtime") or "")
            models[key][0] += s["corrections"]
            models[key][1] += s["correction_labels"]
            models[key][2] += float(s["row"].get("cost") or 0)
    rework = {
        "weekly": [{"week": w, **_rate(c, n)} for w, (c, n) in sorted(weekly.items()) if n],
        "models": sorted(
            [{"model": m, "runtime": r, "spend": round(sp, 6), **_rate(c, n)}
             for (m, r), (c, n, sp) in models.items() if n],
            key=lambda item: -item["samples"])[:20],
        "overall": _rate(sum(s["corrections"] for s in in_window), sum(s["correction_labels"] for s in in_window)),
    }

    cells = []
    for group_name, members in COMPLEXITY_GROUPS:
        for tier in TIERS:
            group = [s for s in in_window if s["complexity"] in members and s["tier"] == tier]
            spend = sum(float(s["row"].get("cost") or 0) for s in group)
            cells.append({"complexity": group_name, "tier": tier, "sessions": len(group), "spend": round(spend, 6),
                          "rework": _rate(sum(s["corrections"] for s in group),
                                          sum(s["correction_labels"] for s in group))})
    rates = [c["rework"]["rate"] for c in cells if c["rework"] and not c["rework"]["few_samples"]]
    median_rate = statistics.median(rates) if rates else None
    for cell in cells:
        cell["flag"] = ""
        if cell["complexity"] == "routine" and cell["tier"] == "premium" and cell["spend"] > 0:
            cell["flag"] = "possible_overspend"
        elif (cell["complexity"] == "complex" and cell["tier"] == "light" and cell["rework"]
              and not cell["rework"]["few_samples"] and median_rate is not None
              and cell["rework"]["rate"] > median_rate):
            cell["flag"] = "possible_false_economy"

    outcomes = _outcomes(in_window)
    effort = _effort(in_window)
    position = _position(in_window)
    model_fit = _model_fit(in_window)
    headlines = _headlines(sessions, all_months, area_names, economics, cells, tier_prices, outcomes, today,
                           effort, corrections_for)

    labeled_sessions = sum(1 for s in in_window if s["area"] != PENDING)
    labeled_turns = sum(s["correction_labels"] for s in in_window)
    later_turns = sum(max(0, s["turns"] - 1) for s in in_window)
    return {
        "months": all_months,
        "areas": segments,
        "allocation": allocation,
        "workstreams": workstreams,
        "economics": economics,
        "rework": rework,
        "right_sizing": {"cells": cells, "tiers_known": bool(tiers), "effort": effort},
        "outcomes": outcomes,
        "position": position,
        "model_fit": model_fit,
        "headlines": headlines,
        "coverage": {
            "sessions": len(in_window), "labeled_sessions": labeled_sessions,
            "turns": later_turns, "labeled_turns": min(labeled_turns, later_turns),
        },
        "filters": {
            "runtimes": sorted(runtime_options),
            "projects": [name for name, _ in project_options.most_common(50)],
        },
    }


def _week_start(day):
    try:
        value = datetime.date.fromisoformat(day)
    except ValueError:
        return ""
    return (value - datetime.timedelta(days=value.weekday())).isoformat()


def _cost(s):
    return float(s["row"].get("cost") or 0)


def _outcomes(in_window):
    buckets = {key: {"outcome": key, "sessions": 0, "spend": 0.0} for key in OUTCOMES}
    rework_cost = total = 0.0
    for s in in_window:
        key = session_outcome(s["turns"], s["sequence"], s["pending"])
        buckets[key]["sessions"] += 1
        buckets[key]["spend"] += _cost(s)
        total += _cost(s)
        rework_cost += _cost(s) * rework_share(s["turns"], s["sequence"])
    rows = []
    for key in OUTCOMES:
        bucket = buckets[key]
        bucket["spend"] = round(bucket["spend"], 6)
        bucket["cost_per_session"] = bucket["spend"] / bucket["sessions"] if bucket["sessions"] else None
        rows.append(bucket)
    labeled = sum(b["sessions"] for b in rows if b["outcome"] in ("accepted", "recovered", "ended_on_pushback"))
    return {
        "buckets": rows,
        "labeled_sessions": labeled,
        "rework_cost": round(rework_cost, 6),
        "rework_share": rework_cost / total if total else None,
        "few_samples": labeled < MIN_RATE_SAMPLES,
    }


def _position(in_window):
    counts = {label: [0, 0] for _lo, _hi, label in POSITION_BUCKETS}
    for s in in_window:
        for ordinal, value in s["sequence"]:
            if value is None:
                continue
            for lo, hi, label in POSITION_BUCKETS:
                if lo <= ordinal <= hi:
                    counts[label][0] += int(value)
                    counts[label][1] += 1
                    break
    return [{"bucket": label, **(_rate(*counts[label]) or {"rate": None, "samples": 0, "few_samples": True})}
            for _lo, _hi, label in POSITION_BUCKETS]


def _model_fit(in_window):
    labeled = [s for s in in_window if s["work_type"] and s["model"]]
    usage = collections.Counter((s["model"], s["row"].get("runtime") or "") for s in labeled)
    models = [key for key, _ in usage.most_common(MAX_FIT_MODELS)]
    work_types = [w for w in WORK_TYPE_ORDER if w != "unclear" and any(s["work_type"] == w for s in labeled)]
    cells = []
    for work_type in work_types:
        row_cells = []
        for model, runtime in models:
            group = [s for s in labeled if s["work_type"] == work_type
                     and (s["model"], s["row"].get("runtime") or "") == (model, runtime)]
            spend = sum(_cost(s) for s in group)
            cell = {"work_type": work_type, "model": model, "runtime": runtime, "sessions": len(group),
                    "cost_per_session": spend / len(group) if group else None,
                    "rework": _rate(sum(s["corrections"] for s in group), sum(s["correction_labels"] for s in group)),
                    "best": False}
            row_cells.append(cell)
        eligible = [c for c in row_cells if c["rework"] and not c["rework"]["few_samples"]]
        if len(eligible) >= 2:
            min(eligible, key=lambda c: (c["rework"]["rate"], c["cost_per_session"] or 0))["best"] = True
        cells.extend(row_cells)
    return {"work_types": work_types,
            "models": [{"model": m, "runtime": r, "sessions": usage[(m, r)]} for m, r in models],
            "cells": cells}


def _month_spend_shares(sessions, month, area_names):
    """Area shares of all spend in the month, matching the allocation chart (Unclear and Pending included)."""
    spend = collections.Counter()
    for s in sessions:
        for day, cost in (s["row"].get("_day_cost") or {}).items():
            if str(day)[:7] == month:
                spend[s["area"]] += float(cost or 0)
    total = sum(spend.values())
    shares = {area: value / total for area, value in spend.items() if area in area_names} if total else {}
    return shares, total


def _month_rework(sessions, month):
    group = [s for s in sessions if s["start_month"] == month]
    return _rate(sum(s["corrections"] for s in group), sum(s["correction_labels"] for s in group))


def _effort(in_window):
    """Complexity × reasoning effort: spend, sessions, pushback; high effort on routine work is flagged."""
    efforts = [e for e in EFFORT_ORDER if any(s["effort"] == e for s in in_window)]
    rows = []
    for group_name, members in COMPLEXITY_GROUPS:
        for effort in efforts:
            group = [s for s in in_window if s["complexity"] in members and s["effort"] == effort]
            spend = sum(_cost(s) for s in group)
            rows.append({"complexity": group_name, "effort": effort, "sessions": len(group),
                         "spend": round(spend, 6),
                         "rework": _rate(sum(s["corrections"] for s in group),
                                         sum(s["correction_labels"] for s in group)),
                         "flag": "possible_overthinking" if group_name == "routine" and effort in HIGH_EFFORTS
                         and spend > 0 else ""})
    return {"efforts": efforts, "cells": rows}


def _resolved(group, corrections_for):
    _attach_sequences([s for s in group if not s["sequence"]], corrections_for)
    return [s for s in group if session_outcome(s["turns"], s["sequence"], s["pending"])
            in ("accepted", "recovered")]


def _headlines(sessions, months, area_names, economics, cells, tier_prices, outcomes, today="",
               effort=None, corrections_for=None):
    """Deterministic, evidence-backed statements; each names the module that supports it."""
    cards = []
    buckets = {b["outcome"]: b for b in outcomes["buckets"]}
    resolved_n = buckets["accepted"]["sessions"] + buckets["recovered"]["sessions"]
    if not outcomes["few_samples"] and resolved_n and buckets["ended_on_pushback"]["sessions"]:
        resolved_cost = (buckets["accepted"]["spend"] + buckets["recovered"]["spend"]) / resolved_n
        ended = buckets["ended_on_pushback"]
        cards.append({"key": "cost_per_resolved", "kind": "neutral", "target": "outcomes",
                      "cost": resolved_cost, "sessions": resolved_n,
                      "ended_cost": ended["spend"] / ended["sessions"]})
    complete = [m for m in months if not (today and today[:7] == m)]
    if len(complete) >= 2:
        last, before = complete[-1], complete[-2]
        spend_last = sum(float(c or 0) for s in sessions for d, c in (s["row"].get("_day_cost") or {}).items()
                         if str(d)[:7] == last)
        spend_before = sum(float(c or 0) for s in sessions for d, c in (s["row"].get("_day_cost") or {}).items()
                           if str(d)[:7] == before)
        resolved_last = len(_resolved([s for s in sessions if s["start_month"] == last], corrections_for))
        resolved_before = len(_resolved([s for s in sessions if s["start_month"] == before], corrections_for))
        if spend_before > 0 and resolved_before >= MIN_RATE_SAMPLES:
            spend_change = spend_last / spend_before - 1
            resolved_change = resolved_last / resolved_before - 1
            if spend_change >= 0.25 and resolved_change <= 0.05:
                cards.append({"key": "value_flat", "kind": "warn", "target": "outcomes", "month": last,
                              "previous_month": before, "spend_change": spend_change,
                              "resolved_change": resolved_change})
    if effort:
        overthinking = [c for c in effort["cells"] if c["flag"] == "possible_overthinking"]
        spend = sum(c["spend"] for c in overthinking)
        if spend >= 1:
            cards.append({"key": "effort_routine", "kind": "warn", "target": "sizing", "spend": spend,
                          "sessions": sum(c["sessions"] for c in overthinking)})
    current = months[-1] if months else ""
    previous = months[-2] if len(months) > 1 else ""
    if current and previous:
        now_shares, now_total = _month_spend_shares(sessions, current, area_names)
        before_shares, before_total = _month_spend_shares(sessions, previous, area_names)
        if now_total and before_total:
            deltas = {a: now_shares.get(a, 0) - before_shares.get(a, 0) for a in set(now_shares) | set(before_shares)}
            area, delta = max(deltas.items(), key=lambda item: abs(item[1]), default=("", 0))
            if area and abs(delta) >= 0.10:
                cards.append({"key": "area_shift", "kind": "neutral", "target": "allocation",
                              "area": area, "month": current, "previous_month": previous,
                              "partial": bool(today and today[:7] == current),
                              "share": now_shares.get(area, 0), "previous_share": before_shares.get(area, 0)})
        now_rate, before_rate = _month_rework(sessions, current), _month_rework(sessions, previous)
        if now_rate and before_rate and not now_rate["few_samples"] and not before_rate["few_samples"]:
            change = now_rate["rate"] - before_rate["rate"]
            if abs(change) >= 0.03:
                cards.append({"key": "pushback_trend", "kind": "good" if change < 0 else "warn",
                              "target": "rework", "month": current, "previous_month": previous,
                              "partial": bool(today and today[:7] == current),
                              "rate": now_rate["rate"], "previous_rate": before_rate["rate"]})
    ended = next(b for b in outcomes["buckets"] if b["outcome"] == "ended_on_pushback")
    if not outcomes["few_samples"] and outcomes["labeled_sessions"]:
        share = ended["sessions"] / outcomes["labeled_sessions"]
        if share >= 0.10:
            cards.append({"key": "ended_on_pushback", "kind": "warn", "target": "outcomes",
                          "share": share, "sessions": ended["sessions"], "spend": ended["spend"]})
    overspend = next((c for c in cells if c["flag"] == "possible_overspend"), None)
    if overspend and overspend["spend"] >= 1 and tier_prices.get("premium") and tier_prices.get("standard"):
        ratio = tier_prices["standard"] / tier_prices["premium"]
        cards.append({"key": "right_size", "kind": "warn", "target": "sizing", "spend": overspend["spend"],
                      "sessions": overspend["sessions"], "estimate": round(overspend["spend"] * ratio, 6)})
    typed = [e for e in economics if e["work_type"] not in ("unclear", "other") and e["sessions"] >= 3]
    if len(typed) >= 2:
        costliest = max(typed, key=lambda e: e["cost_per_session"])
        median = statistics.median(e["cost_per_session"] for e in typed)
        if median and costliest["cost_per_session"] >= 1.5 * median:
            cards.append({"key": "costliest_work", "kind": "neutral", "target": "economics",
                          "work_type": costliest["work_type"], "cost_per_session": costliest["cost_per_session"],
                          "multiple": costliest["cost_per_session"] / median})
    if rework_value := outcomes.get("rework_share"):
        if rework_value >= 0.10 and not outcomes["few_samples"]:
            cards.append({"key": "rework_cost", "kind": "warn", "target": "outcomes",
                          "spend": outcomes["rework_cost"], "share": rework_value})
    order = {"warn": 0, "good": 1, "neutral": 2}
    cards.sort(key=lambda card: order[card["kind"]])
    return cards[:MAX_HEADLINES]


MAX_DRILL_SESSIONS = 50
DRILL_FILTERS = ("month", "area", "work_type", "complexity", "tier", "effort", "outcome", "model", "model_runtime")


def find_sessions(rows, labels, key_for, areas, output_price, filters, months=6, runtime="", project="",
                  corrections_for=None, pending_keys=None, limit=MAX_DRILL_SESSIONS):
    """Sessions behind one Work module cell, ranked by spend; same windowing and labels as the aggregates.

    ``month`` selects sessions active in that month (as allocation and workstreams count them);
    every other filter applies to sessions started in the selected period (as the other modules do).
    """
    area_names = [a["name"] for a in areas]
    tiers = price_tiers(rows, output_price)
    sessions, _runtimes, _projects = _prepare_sessions(
        rows, labels, key_for, area_names, tiers, runtime, project, pending_keys)
    all_months = _window_months(sessions, months)
    month = filters.get("month") or ""
    if month:
        candidates = [s for s in sessions if month in all_months and month in _session_months(s)]
    else:
        window = set(all_months)
        candidates = [s for s in sessions if s["start_month"] in window]
    groups = dict(COMPLEXITY_GROUPS)
    matched = []
    for s in candidates:
        if filters.get("area") and s["area"] != filters["area"]:
            continue
        if filters.get("work_type") and (s["work_type"] or "") != filters["work_type"]:
            continue
        if filters.get("complexity") and s["complexity"] not in groups.get(filters["complexity"], ()):
            continue
        if filters.get("tier") and s["tier"] != filters["tier"]:
            continue
        if filters.get("effort") and s["effort"] != filters["effort"]:
            continue
        if filters.get("model") and (s["model"] != filters["model"]
                                     or (s["row"].get("runtime") or "") != filters.get("model_runtime", "")):
            continue
        matched.append(s)
    _attach_sequences(matched, corrections_for)
    if filters.get("outcome"):
        matched = [s for s in matched
                   if session_outcome(s["turns"], s["sequence"], s["pending"]) == filters["outcome"]]
    matched.sort(key=lambda s: (-_cost(s), s["row"].get("last") or ""))
    return {
        "total": len(matched),
        "spend": round(sum(_cost(s) for s in matched), 6),
        "truncated": len(matched) > limit,
        "sessions": [{
            "id": s["row"].get("id") or "",
            "title": str(s["row"].get("session_name") or s["row"].get("title") or "")[:90],
            "runtime": s["row"].get("runtime") or "",
            "project": s["row"].get("project") or "",
            "start": s["row"].get("start") or "",
            "last": s["row"].get("last") or "",
            "cost": round(_cost(s), 6),
            "turns": s["turns"],
            "model": s["model"],
            "area": s["area"],
            "work_type": s["work_type"],
            "complexity": s["complexity"],
            "outcome": session_outcome(s["turns"], s["sequence"], s["pending"]),
            "corrections": s["corrections"],
            "labeled_turns": s["correction_labels"],
        } for s in matched[:limit]],
    }
