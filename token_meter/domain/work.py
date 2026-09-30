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
    return any(
        isinstance(record, dict) and record.get("parent_id")
        for record in row.get("_agent_records") or ()
    )


def turn_days(row):
    events = (row.get("_language_signal_events") or {}).get("positive") or []
    return [str(event.get("day") or "") for event in events]


def primary_model(row):
    stats = [s for s in row.get("model_stats") or [] if isinstance(s, dict) and s.get("model")]
    if stats:
        return max(stats, key=lambda s: (float(s.get("cost") or 0), int(s.get("tokens") or 0)))["model"]
    models = row.get("models") or []
    return models[0] if models else ""


def price_tiers(rows, output_price):
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
        return {}
    if len(distinct) == 1:
        return {key: "standard" for key in prices}

    def tier(price):
        position = distinct.index(price) / (len(distinct) - 1)
        return "light" if position < 1 / 3 else "premium" if position > 2 / 3 else "standard"

    return {key: tier(price) for key, price in prices.items()}


def _month(day):
    return day[:7] if len(day) >= 7 else ""


def _rate(corrections, samples):
    if samples <= 0:
        return None
    return {"rate": corrections / samples, "samples": samples, "few_samples": samples < MIN_RATE_SAMPLES}


def build_work_insights(rows, labels, key_for, areas, output_price, months=6,
                        runtime="", project="", today=""):
    """Aggregate labeled sessions. ``months`` 0 means all history."""
    area_names = [a["name"] for a in areas]
    tiers = price_tiers(rows, output_price)
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
        })

    all_months = sorted({m for s in sessions for m in [s["start_month"], *map(_month, s["days"]),
                                                      *map(_month, (s["row"].get("_day_cost") or {}))] if m})
    if months:
        all_months = all_months[-months:]
    all_months = all_months[-MAX_MONTHS:]
    month_set = set(all_months)
    segments = area_names + [UNCLEAR, PENDING]

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
        "right_sizing": {"cells": cells, "tiers_known": bool(tiers)},
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
