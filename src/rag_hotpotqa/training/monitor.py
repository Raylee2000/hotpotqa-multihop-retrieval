"""Live training curves without plotting dependencies: CSV + SVG + auto-refreshing HTML.

`render(run_dir)` rebuilds everything from the append-only logs (train_log.jsonl, val_loss.jsonl,
selection.jsonl), so it is resume-safe (the last record of a step wins) and can also run from a separate
process. Files are replaced atomically; open <run_dir>/live/index.html in a browser.
"""
from __future__ import annotations

import csv
import html
import io
import json
import os
from pathlib import Path

W, H, PAD = 560, 260, 44
COLORS = ("#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c", "#495057")


def _records(path: Path, key: str = "step") -> list[dict]:
    if not path.exists():
        return []
    out: dict[int, dict] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line being written right now
            continue
        if key in r:
            out[r[key]] = r
    return [out[k] for k in sorted(out)]


def _ema(xs: list[float], a: float = 0.98) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _csv(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _panel(title: str, series: list[tuple[str, list[float], list[float], float]], y_label: str) -> str:
    """One SVG panel; series = (name, xs, ys, stroke opacity)."""
    pts = [(x, y) for _, xs, ys, _ in series for x, y in zip(xs, ys)]
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">'
    head += f'<rect width="{W}" height="{H}" fill="#fff"/><text x="{PAD}" y="16" font-size="13" font-weight="bold">{html.escape(title)}</text>'
    if not pts:
        return head + f'<text x="{PAD}" y="{H // 2}" fill="#868e96">no data yet</text></svg>'
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    x1, y1 = (x1 if x1 > x0 else x0 + 1), (y1 if y1 > y0 else y0 + 1e-6)
    sx = lambda x: PAD + (x - x0) / (x1 - x0) * (W - PAD - 12)  # noqa: E731
    sy = lambda y: H - 28 - (y - y0) / (y1 - y0) * (H - 56)  # noqa: E731
    out = [head, f'<line x1="{PAD}" y1="{H - 28}" x2="{W - 12}" y2="{H - 28}" stroke="#adb5bd"/>',
           f'<line x1="{PAD}" y1="28" x2="{PAD}" y2="{H - 28}" stroke="#adb5bd"/>']
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        out.append(f'<text x="{PAD - 4}" y="{sy(y) + 4:.1f}" text-anchor="end" fill="#495057">{y:.3g}</text>')
        x = x0 + (x1 - x0) * i / 4
        out.append(f'<text x="{sx(x):.1f}" y="{H - 12}" text-anchor="middle" fill="#495057">{x:.0f}</text>')
    out.append(f'<text x="{W - 12}" y="{H - 12}" text-anchor="end" fill="#868e96">step</text>')
    out.append(f'<text x="8" y="24" fill="#868e96">{html.escape(y_label)}</text>')
    for i, (name, xs, ys, op) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        if len(xs) == 1:
            out.append(f'<circle cx="{sx(xs[0]):.1f}" cy="{sy(ys[0]):.1f}" r="3" fill="{c}"/>')
        elif xs:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
            out.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-opacity="{op}" stroke-width="1.5"/>')
            if len(xs) <= 30:
                out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{c}"/>' for x, y in zip(xs, ys)]
        out.append(f'<text x="{W - 16}" y="{34 + 14 * i}" text-anchor="end" fill="{c}">{html.escape(name)}'
                   f'{"" if not ys else f" = {ys[-1]:.4g}"}</text>')
    return "".join(out) + "</svg>"


def render(run_dir: Path, title: str = "") -> dict:
    run_dir = Path(run_dir)
    live = run_dir / "live"
    live.mkdir(exist_ok=True)
    tr = _records(run_dir / "train_log.jsonl")
    va = _records(run_dir / "val_loss.jsonl")
    se = _records(run_dir / "selection.jsonl")
    xs = [r["step"] for r in tr]
    loss = [r["loss"] for r in tr]
    ema = _ema(loss)
    for r, e in zip(tr, ema):
        r["loss_ema"] = round(e, 5)
    _write(run_dir / "train_metrics.csv", _csv(tr, ["step", "epoch", "loss", "loss_ema", "grad_norm", "lr", "pos_cos", "n_docs", "masked", "seconds"]))
    _write(run_dir / "val_metrics.csv", _csv(va, ["step", "val_loss", "n_examples", "seconds"]))
    sel_rows = [{"step": r["step"], **r["metrics"]} for r in se]
    sel_cols = ["step"] + (list(se[0]["metrics"]) if se else [])
    _write(run_dir / "selection_metrics.csv", _csv(sel_rows, sel_cols))
    panels = [
        _panel("Training loss (InfoNCE)", [("loss", xs, loss, 0.25), ("EMA 0.98", xs, ema, 1.0)], "loss"),
        _panel("Validation loss (fixed V-select examples, no grad)", [("val loss", [r["step"] for r in va], [r["val_loss"] for r in va], 1.0)], "loss"),
        _panel("Validation retrieval metrics (selection subset)",
               [(k, [r["step"] for r in sel_rows], [r[k] for r in sel_rows], 1.0)
                for k in ("score", "O.fh_mrr", "O.fh_all_gold@2", "E1.AtoB.recall@1") if sel_rows and k in sel_rows[0]], "metric"),
        _panel("Gradient norm (before clipping)", [("grad norm", xs, [r["grad_norm"] for r in tr], 0.4),
                                                   ("EMA", xs, _ema([r["grad_norm"] for r in tr]), 1.0)], "norm"),
    ]
    _write(live / "curves.svg", "".join(panels))
    for i, p in enumerate(panels):
        _write(live / f"panel{i}.svg", p)
    last = tr[-1] if tr else {}
    info = {"last_step": last.get("step"), "last_loss_ema": last.get("loss_ema"), "val_points": len(va), "selection_points": len(se)}
    page = ("<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
            f"<title>{html.escape(title or run_dir.name)} live</title>"
            "<style>body{font-family:sans-serif;margin:16px;background:#f8f9fa}img{background:#fff;margin:6px;border:1px solid #dee2e6}</style>"
            f"</head><body><h2>{html.escape(title or run_dir.name)}</h2><p>{html.escape(json.dumps(info))} (auto-refresh 30 s)</p>"
            + "".join(f"<img src='panel{i}.svg'>" for i in range(len(panels))) + "</body></html>")
    _write(live / "index.html", page)
    return info
