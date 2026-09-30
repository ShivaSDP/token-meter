"""Runtime-neutral work insights: allocation, outcomes, work-type economics, rework, right-sizing.

Pure functions over cached session summary rows and content-free classifier labels.
"""

import collections
import datetime
import statistics

UNCLEAR = "Unclear"
PENDING = "Pending"
MIN_RATE_SAMPLES = 20
MAX_MONTHS = 24
COMPLEXITY_GROUPS = (
    ("routine", ("routine",)),
    ("everyday", ("everyday",)),
    ("complex", ("complex", "high_impact")),
)
TIERS = ("light", "standard", "premium")
WORK_TYPE_ORDER = ("debug", "feature", "refactor", "docs", "explore", "review", "ops", "other", "unclear")


def work_identity(row):
    """Per-trace identity: one session id can span several rollout files (resumes, forks)."""
    return f"{row.get('id') or ''}\0{row.get('path') or ''}"


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
MAX_HEADLINES = 3
MAX_TREND_MODELS = 6
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max", "ultra")
HIGH_EFFORTS = ("xhigh", "max", "ultra")


def session_outcome(turns, sequence, pending=False, labeled=False):
    """Classify a session from its ordered follow-up pushback labels.

    A session is classified only once it has no queued or backlogged work (``pending``);
    sessions whose labels are all low-confidence are Unclear rather than counted as accepted.
    A ``labeled`` session with no pending work and no follow-up labels had no classifiable
    follow-ups (greetings, image- or wrapper-only turns), so it is a single shot; a session
    with no labels at all is still waiting for the classifier.
    """
    if turns <= 1:
        return "single_shot"
    if pending:
        return "pending"
    if not sequence:
        return "single_shot" if labeled else "pending"
    confident = [value for _ordinal, value in sequence if value is not None]
    if not confident:
        return "unclear"
    if not any(confident):
        return "accepted"
    return "ended_on_pushback" if confident[-1] else "recovered"


def _outcome(s):
    return session_outcome(s["turns"], s["sequence"], s["pending"], labeled=bool(s["entry"]))


def _month_shift(month, delta):
    year, number = divmod(int(month[:4]) * 12 + int(month[5:7]) - 1 + delta, 12)
    return f"{year:04d}-{number + 1:02d}"


def build_work_insights(rows, labels, key_for, areas, output_price, months=6,
                        runtime="", project="", today="", corrections_for=None, pending_keys=None):
    """Aggregate labeled sessions. ``months`` 0 means all history."""
    area_names = [a["name"] for a in areas]
    tiers, tier_prices = price_tiers(rows, output_price, with_prices=True)
    sessions, runtime_options, project_options = _prepare_sessions(
        rows, labels, key_for, area_names, tiers, runtime, project, pending_keys)
    all_months = _window_months(sessions, months)
    month_set = set(all_months)
    # The comparison period is the same number of calendar months just before the current window.
    previous = {_month_shift(all_months[0], -step) for step in range(1, months + 1)} if months and all_months else set()
    segments = area_names + [UNCLEAR, PENDING]
    return _aggregate(sessions, all_months, month_set, segments, area_names, tiers, tier_prices,
                      runtime_options, project_options, today, corrections_for, previous)


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
        entry = labels.get(key_for(work_identity(row))) or {}
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
            "pending": bool(pending_keys) and key_for(work_identity(row)) in pending_keys,
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
            s["sequence"] = corrections_for(work_identity(s["row"]), s["turns"])


def _aggregate(sessions, all_months, month_set, segments, area_names, tiers, tier_prices,
               runtime_options, project_options, today, corrections_for, previous_months=frozenset()):
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

    in_window = [s for s in sessions if s["start_month"] in month_set]
    _attach_sequences(in_window, corrections_for)
    for s in in_window:
        bucket = by_month.get(s["start_month"])
        if bucket is not None:
            outcome = _outcome(s)
            bucket.setdefault("outcomes", {}).setdefault(outcome, 0)
            bucket["outcomes"][outcome] += 1
            bucket.setdefault("outcome_spend", {}).setdefault(outcome, 0.0)
            bucket["outcome_spend"][outcome] = round(bucket["outcome_spend"][outcome] + _cost(s), 6)
    earlier = [s for s in sessions if s["start_month"] in previous_months]
    _attach_sequences(earlier, corrections_for)
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
        outcomes_here = [_outcome(s) for s in group]
        judged = sum(o in ("accepted", "recovered", "ended_on_pushback") for o in outcomes_here)
        resolved = [s for s, o in zip(group, outcomes_here) if o in ("accepted", "recovered")]
        economics.append({
            "work_type": work_type, "sessions": len(group), "spend": round(spend, 6),
            "cost_per_session": spend / len(group),
            "judged_sessions": judged,
            "resolved_sessions": len(resolved),
            "resolved_rate": len(resolved) / judged if judged else None,
            "cost_per_resolved": (sum(_cost(s) for s in resolved) / len(resolved)) if resolved else None,
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
    top_models = sorted(models, key=lambda key: -models[key][1])[:MAX_TREND_MODELS]
    by_model = {key: collections.defaultdict(lambda: [0, 0]) for key in top_models}
    for s in in_window:
        key = (s["model"], s["row"].get("runtime") or "")
        if key in by_model and s["correction_labels"] and s["days"]:
            week = _week_start(s["days"][0])
            by_model[key][week][0] += s["corrections"]
            by_model[key][week][1] += s["correction_labels"]
    rework = {
        "weekly": [{"week": w, **_rate(c, n)} for w, (c, n) in sorted(weekly.items()) if n],
        "weekly_by_model": [{"model": m, "runtime": r,
                             "weeks": [{"week": w, **_rate(c, n)} for w, (c, n) in sorted(by_model[(m, r)].items()) if n]}
                            for m, r in top_models],
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

    effort = _effort(in_window)
    kpis = {"current": _kpis(in_window), "previous": _kpis(earlier) if earlier else None,
            "previous_months": sorted(previous_months)}
    position = _position(in_window)
    model_fit = _model_fit(in_window)
    opportunities = _opportunities(cells, effort, tier_prices)
    headlines = _headlines(sessions, all_months, area_names, economics, tier_prices, position, opportunities,
                           today, corrections_for)

    labeled_sessions = sum(1 for s in in_window if s["area"] != PENDING)
    labeled_turns = sum(s["correction_labels"] for s in in_window)
    later_turns = sum(max(0, s["turns"] - 1) for s in in_window)
    return {
        "months": all_months,
        "areas": segments,
        "allocation": allocation,
        "economics": economics,
        "rework": rework,
        "right_sizing": {"cells": cells, "tiers_known": bool(tiers), "effort": effort},
        "kpis": kpis,
        "model_fit": model_fit,
        "choices": _choices(model_fit),
        "opportunities": opportunities,
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


def _kpis(group):
    """Period KPIs: resolved share, pushback rate, cost per resolved session, spend after first pushback."""
    counts = collections.Counter()
    spend = collections.Counter()
    for s in group:
        outcome = _outcome(s)
        counts[outcome] += 1
        spend[outcome] += _cost(s)
    judged = counts["accepted"] + counts["recovered"] + counts["ended_on_pushback"]
    resolved = counts["accepted"] + counts["recovered"]
    total = sum(spend.values())
    return {
        "sessions": len(group),
        "judged_sessions": judged,
        "resolved_rate": resolved / judged if judged else None,
        "pushback": _rate(sum(s["corrections"] for s in group), sum(s["correction_labels"] for s in group)),
        "cost_per_resolved": (spend["accepted"] + spend["recovered"]) / resolved if resolved else None,
        "ended_spend": round(spend["ended_on_pushback"], 6),
        "ended_share": spend["ended_on_pushback"] / total if total else None,
        "spend": round(total, 6),
        "few_samples": judged < MIN_RATE_SAMPLES,
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
    return [s for s in group if _outcome(s)
            in ("accepted", "recovered")]


def _headlines(sessions, months, area_names, economics, tier_prices, position, opportunities, today="",
               corrections_for=None):
    """Up to three statements about change or opportunity; each links to the module that supports it.

    KPIs already state the period's levels, so cards only report movement, the largest opportunity,
    and patterns that are otherwise easy to miss.
    """
    cards = []
    current = months[-1] if months else ""
    previous = months[-2] if len(months) > 1 else ""
    partial = bool(today and today[:7] == current)
    complete = [m for m in months if not (today and today[:7] == m)]
    if len(complete) >= 2:
        last, before = complete[-1], complete[-2]

        def month_spend(month):
            return sum(float(c or 0) for s in sessions for d, c in (s["row"].get("_day_cost") or {}).items()
                       if str(d)[:7] == month)

        resolved_last = len(_resolved([s for s in sessions if s["start_month"] == last], corrections_for))
        resolved_before = len(_resolved([s for s in sessions if s["start_month"] == before], corrections_for))
        spend_last, spend_before = month_spend(last), month_spend(before)
        if spend_before > 0 and resolved_before >= MIN_RATE_SAMPLES:
            spend_change, resolved_change = spend_last / spend_before - 1, resolved_last / resolved_before - 1
            if spend_change >= 0.25 and resolved_change <= 0.05:
                cards.append({"key": "value_flat", "kind": "warn", "target": "allocation", "month": last,
                              "previous_month": before, "spend_change": spend_change,
                              "resolved_change": resolved_change})
    if current and previous:
        now_rate, before_rate = _month_rework(sessions, current), _month_rework(sessions, previous)
        if now_rate and before_rate and not now_rate["few_samples"] and not before_rate["few_samples"]:
            change = now_rate["rate"] - before_rate["rate"]
            if abs(change) >= 0.03:
                cards.append({"key": "pushback_trend", "kind": "good" if change < 0 else "warn",
                              "target": "rework", "month": current, "previous_month": previous,
                              "partial": partial, "rate": now_rate["rate"], "previous_rate": before_rate["rate"]})
    if opportunities and opportunities[0]["spend"] >= 1:
        top = opportunities[0]
        cards.append({"key": "top_opportunity", "kind": "warn", "target": "sizing",
                      **{k: v for k, v in top.items() if k != "kind"}, "opportunity": top["kind"]})
    early = next((p for p in position if p["bucket"] == "1–2"), None)
    late = next((p for p in position if p["bucket"] in ("11–20", "21+") and p["rate"] is not None
                 and not p["few_samples"]), None)
    if early and late and early["rate"] and not early["few_samples"] and late["rate"] >= 1.5 * early["rate"]:
        cards.append({"key": "long_sessions_drift", "kind": "warn", "target": "rework",
                      "bucket": late["bucket"], "rate": late["rate"], "early_rate": early["rate"]})
    if current and previous:
        now_shares, now_total = _month_spend_shares(sessions, current, area_names)
        before_shares, before_total = _month_spend_shares(sessions, previous, area_names)
        if now_total and before_total:
            deltas = {a: now_shares.get(a, 0) - before_shares.get(a, 0) for a in set(now_shares) | set(before_shares)}
            area, delta = max(deltas.items(), key=lambda item: abs(item[1]), default=("", 0))
            if area and abs(delta) >= 0.10:
                cards.append({"key": "area_shift", "kind": "neutral", "target": "allocation", "area": area,
                              "month": current, "previous_month": previous, "partial": partial,
                              "share": now_shares.get(area, 0), "previous_share": before_shares.get(area, 0)})
    judged = [e for e in economics if e["work_type"] not in ("unclear", "other")
              and e.get("cost_per_resolved") is not None and e["resolved_sessions"] >= 3]
    if len(judged) >= 2:
        costliest = max(judged, key=lambda e: e["cost_per_resolved"])
        median = statistics.median(e["cost_per_resolved"] for e in judged)
        if median and costliest["cost_per_resolved"] >= 2 * median:
            cards.append({"key": "costliest_work", "kind": "neutral", "target": "economics",
                          "work_type": costliest["work_type"], "cost_per_resolved": costliest["cost_per_resolved"],
                          "multiple": costliest["cost_per_resolved"] / median})
    order = {"warn": 0, "good": 1, "neutral": 2}
    cards.sort(key=lambda card: order[card["kind"]])
    return cards[:MAX_HEADLINES]


def _choices(model_fit):
    """Per work type: the least-pushback model vs the model used most, when both have enough evidence."""
    rows = []
    for work_type in model_fit["work_types"]:
        cells = [c for c in model_fit["cells"] if c["work_type"] == work_type and c["sessions"]]
        eligible = [c for c in cells if c["rework"] and not c["rework"]["few_samples"]]
        if len(eligible) < 2:
            continue
        best = min(eligible, key=lambda c: (c["rework"]["rate"], c["cost_per_session"] or 0))
        used = max(cells, key=lambda c: c["sessions"])
        if best is used or not used["rework"] or used["rework"]["few_samples"]:
            continue  # The usual model needs the same labeled evidence before it is compared.
        if best["rework"]["rate"] + 0.03 > used["rework"]["rate"]:
            continue
        rows.append({"work_type": work_type, "best": best, "used": used,
                     "gap": used["rework"]["rate"] - best["rework"]["rate"]})
    return sorted(rows, key=lambda row: -row["gap"])


def _opportunities(cells, effort, tier_prices):
    """Flagged right-sizing cells as one ranked list, with a savings estimate where prices allow."""
    rows = []
    ratio = (tier_prices["standard"] / tier_prices["premium"]
             if tier_prices.get("premium") and tier_prices.get("standard") else None)
    for cell in cells:
        if cell["flag"] == "possible_overspend":
            rows.append({"kind": "premium_routine", "complexity": cell["complexity"], "tier": cell["tier"],
                         "sessions": cell["sessions"], "spend": cell["spend"],
                         "estimate": round(cell["spend"] * ratio, 6) if ratio else None})
        elif cell["flag"] == "possible_false_economy":
            rows.append({"kind": "light_complex", "complexity": cell["complexity"], "tier": cell["tier"],
                         "sessions": cell["sessions"], "spend": cell["spend"], "rework": cell["rework"],
                         "estimate": None})
    for cell in (effort or {}).get("cells", []):
        if cell["flag"] == "possible_overthinking":
            rows.append({"kind": "effort_routine", "complexity": cell["complexity"], "effort": cell["effort"],
                         "sessions": cell["sessions"], "spend": cell["spend"], "estimate": None})
    return sorted(rows, key=lambda row: -row["spend"])


MAX_DRILL_SESSIONS = 50
MAX_DRILL_IDS = 2000
DRILL_FILTERS = ("month", "start_month", "area", "work_type", "complexity", "tier", "effort", "outcome",
                 "model", "model_runtime")


def find_sessions(rows, labels, key_for, areas, output_price, filters, months=6, runtime="", project="",
                  corrections_for=None, pending_keys=None, limit=MAX_DRILL_SESSIONS, ids_only=False):
    """Sessions behind one Work module cell, ranked by spend; same windowing and labels as the aggregates.

    ``month`` selects sessions active in that month (as the allocation counts turns and spend);
    ``start_month`` selects sessions started in that month (as outcomes and session counts do);
    every other filter applies to sessions started in the selected period.
    """
    area_names = [a["name"] for a in areas]
    tiers = price_tiers(rows, output_price)
    sessions, _runtimes, _projects = _prepare_sessions(
        rows, labels, key_for, area_names, tiers, runtime, project, pending_keys)
    all_months = _window_months(sessions, months)
    month = filters.get("month") or ""
    start_month = filters.get("start_month") or ""
    if start_month:
        candidates = [s for s in sessions if start_month in all_months and s["start_month"] == start_month]
    elif month:
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
                   if _outcome(s) == filters["outcome"]]
    matched.sort(key=lambda s: (-_cost(s), s["row"].get("last") or ""))
    if ids_only:
        return {"total": len(matched), "spend": round(sum(_cost(s) for s in matched), 6),
                "truncated": len(matched) > MAX_DRILL_IDS,
                "ids": list(dict.fromkeys(s["row"].get("id") or "" for s in matched))[:MAX_DRILL_IDS]}
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
            "outcome": _outcome(s),
            "corrections": s["corrections"],
            "labeled_turns": s["correction_labels"],
        } for s in matched[:limit]],
    }
