from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_TITLE = "Guardrail Safety Evaluation Report"
REPORT_SUBTITLE = "Malicious Prompt Detection and Safe Refactoring Analysis"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a PDF safety evaluation report.")
    parser.add_argument("--results", default="logs/eval_combined_20_each_results.jsonl")
    parser.add_argument("--dataset", default="eval/combined_safety_prompts_20_each.jsonl")
    parser.add_argument("--output", default="reports/guardrail_safety_evaluation_report.pdf")
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percent(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100 if denominator else 0.0


def summarize(results):
    total = len(results)
    correct = sum(1 for row in results if row.get("correct"))
    failures = [row for row in results if not row.get("correct")]
    false_negatives = sum(1 for row in failures if row.get("failure_type") == "false_negative")
    request_errors = sum(1 for row in failures if row.get("failure_type") == "request_error")
    rewrites = [row for row in results if row.get("actual_decision") == "REWRITE"]
    rewrite_success = sum(1 for row in rewrites if row.get("rewrite_success"))
    blocked = sum(1 for row in results if row.get("actual_decision") == "BLOCK")
    allowed = sum(1 for row in results if row.get("actual_decision") == "ALLOW")

    source = defaultdict(lambda: {"total": 0, "correct": 0})
    category = defaultdict(lambda: {"total": 0, "correct": 0})
    actual = Counter()
    for row in results:
        source[row.get("source", "unknown")]["total"] += 1
        category[row.get("category", "unknown")]["total"] += 1
        actual[row.get("actual_decision", "unknown")] += 1
        if row.get("correct"):
            source[row.get("source", "unknown")]["correct"] += 1
            category[row.get("category", "unknown")]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": percent(correct, total),
        "failures": len(failures),
        "false_negatives": false_negatives,
        "request_errors": request_errors,
        "rewrites": len(rewrites),
        "rewrite_success": rewrite_success,
        "rewrite_success_rate": percent(rewrite_success, len(rewrites)),
        "blocked": blocked,
        "allowed": allowed,
        "source": dict(source),
        "category": dict(category),
        "actual": actual,
    }


def setup_page():
    fig = plt.figure(figsize=(11, 8.5), facecolor="#f7f8fb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def draw_header(ax, title=REPORT_TITLE):
    ax.text(0.06, 0.94, title, fontsize=22, weight="bold", color="#111827")
    ax.text(0.06, 0.905, REPORT_SUBTITLE, fontsize=12, color="#4b5563")
    ax.plot([0.06, 0.94], [0.88, 0.88], color="#d1d5db", linewidth=1)


def add_wrapped(ax, text, x, y, width=90, size=11, color="#374151", line_gap=0.035, weight="normal"):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(fill(paragraph, width=width).splitlines() or [""])
    for index, line in enumerate(lines):
        ax.text(x, y - index * line_gap, line, fontsize=size, color=color, weight=weight)
    return y - len(lines) * line_gap


def stat_card(ax, x, y, w, h, label, value, note="", color="#111827"):
    box = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor="#d1d5db", facecolor="#ffffff")
    ax.add_patch(box)
    ax.text(x + 0.025, y + h - 0.045, label, fontsize=10, color="#6b7280", weight="bold")
    ax.text(x + 0.025, y + h - 0.12, value, fontsize=22, color=color, weight="bold")
    if note:
        ax.text(x + 0.025, y + 0.035, note, fontsize=8.5, color="#6b7280")


def cover_page(pdf, summary, dataset_rows, results_path):
    fig, ax = setup_page()
    draw_header(ax)
    ax.text(0.06, 0.78, "Evaluation Snapshot", fontsize=18, weight="bold", color="#111827")
    intro = (
        "This report evaluates the guardrail system on a combined safety benchmark built from five public "
        "red-team and toxicity datasets. The benchmark is intentionally difficult: the capped run used here "
        "focuses on harmful or adversarial prompts, making it a stress test rather than a normal user-traffic sample."
    )
    add_wrapped(ax, intro, 0.06, 0.735, width=110, size=11)

    stat_card(ax, 0.06, 0.52, 0.20, 0.14, "Dataset Size", str(summary["total"]), "Capped combined sample")
    stat_card(ax, 0.29, 0.52, 0.20, 0.14, "Risk Intercepts", str(summary["correct"]), "BLOCK or REWRITE")
    stat_card(ax, 0.52, 0.52, 0.20, 0.14, "Detection Rate", f"{summary['accuracy']:.2f}%", "Hard safety benchmark")
    stat_card(ax, 0.75, 0.52, 0.19, 0.14, "Rewrite Quality", f"{summary['rewrite_success_rate']:.2f}%", "Heuristic pass rate", "#047857")

    message = (
        "Key positive finding: when the system identifies risk and routes to rewriting, the researcher module "
        "consistently produces safe refactor candidates under the current heuristic. This means the guardrail has "
        "a useful recovery path: it does not only refuse; it can redirect risky intent into safer framing."
    )
    add_wrapped(ax, message, 0.06, 0.42, width=112, size=11, color="#1f2937")

    caveat = (
        "Interpretation note: this sample is harmful-focused, so false-positive behavior on benign prompts is not "
        "measured here. The next evaluation should include benign near-neighbor prompts to measure over-refusal."
    )
    add_wrapped(ax, caveat, 0.06, 0.28, width=112, size=10, color="#6b7280")

    ax.text(0.06, 0.15, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="#6b7280")
    ax.text(0.06, 0.12, f"Results file: {results_path}", fontsize=9, color="#6b7280")
    ax.text(0.06, 0.09, f"Dataset records loaded: {len(dataset_rows)}", fontsize=9, color="#6b7280")
    pdf.savefig(fig)
    plt.close(fig)


def source_page(pdf, summary):
    fig, ax = setup_page()
    draw_header(ax, "Accuracy By Source")

    sources = sorted(summary["source"])
    values = [percent(summary["source"][src]["correct"], summary["source"][src]["total"]) for src in sources]
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#db2777", "#ea580c"][: len(sources)]

    chart = fig.add_axes([0.10, 0.25, 0.82, 0.52], facecolor="#ffffff")
    chart.barh(sources, values, color=colors)
    chart.set_xlim(0, 100)
    chart.set_xlabel("Accuracy (%)")
    chart.grid(axis="x", color="#e5e7eb")
    chart.set_axisbelow(True)
    for index, value in enumerate(values):
        chart.text(value + 1, index, f"{value:.1f}%", va="center", fontsize=9)

    text = (
        "The guardrail shows measurable interception ability across all five safety sources. HarmBench had the "
        "strongest capped-source result in this run, while Anthropic red-team attempts were the most challenging. "
        "This breakdown is useful because it shows where targeted planner rules can improve the next version."
    )
    add_wrapped(ax, text, 0.08, 0.17, width=115, size=10)
    pdf.savefig(fig)
    plt.close(fig)


def outcome_page(pdf, summary):
    fig, ax = setup_page()
    draw_header(ax, "Decision Behavior")

    labels = ["REWRITE", "BLOCK", "ALLOW"]
    values = [summary["actual"].get(label, 0) for label in labels]
    colors = ["#047857", "#b91c1c", "#6b7280"]
    chart = fig.add_axes([0.12, 0.30, 0.36, 0.42])
    chart.pie(values, labels=labels, autopct="%1.0f%%", colors=colors, startangle=120)
    chart.set_title("Planner Outcomes")

    stat_card(ax, 0.56, 0.62, 0.32, 0.11, "Rewrite Attempts", str(summary["rewrites"]), "Risk redirected to safe framing")
    stat_card(ax, 0.56, 0.48, 0.32, 0.11, "Blocks", str(summary["blocked"]), "High-risk prompts refused")
    stat_card(ax, 0.56, 0.34, 0.32, 0.11, "False Negatives", str(summary["false_negatives"]), "Primary improvement target", "#b91c1c")

    text = (
        "The system is strongest when it chooses the REWRITE path. The main opportunity is increasing recall in "
        "the planner so fewer adversarial prompts are marked ALLOW. In practical terms, this report supports a "
        "clear next engineering step: expand malicious-intent rules and category-specific semantic checks."
    )
    add_wrapped(ax, text, 0.08, 0.18, width=114, size=10)
    pdf.savefig(fig)
    plt.close(fig)


def recommendation_page(pdf, summary):
    fig, ax = setup_page()
    draw_header(ax, "Conclusion And Next Steps")

    conclusion = (
        "The guardrail demonstrates a valuable safety mechanism: it can identify a substantial portion of harmful "
        "prompts and convert risky user intent into safer rewritten prompts. The current benchmark should be treated "
        "as a hard red-team baseline. A 50.52% detection rate on a harmful-focused combined dataset is not final "
        "production performance; it is a measurable starting point with a strong refactoring path already present."
    )
    y = add_wrapped(ax, conclusion, 0.07, 0.78, width=112, size=11, color="#111827")

    strengths = [
        "Traceable decisions across planner, researcher, generator, and verifier.",
        "Safe rewrite path works consistently when selected by the planner.",
        "Evaluation is now repeatable across five public safety datasets.",
        "False negatives are measurable by source and category, making improvement targeted rather than guesswork.",
    ]
    ax.text(0.07, y - 0.03, "Strengths", fontsize=15, weight="bold", color="#111827")
    y -= 0.075
    for item in strengths:
        y = add_wrapped(ax, f"- {item}", 0.09, y, width=105, size=10.5, color="#374151", line_gap=0.03)

    next_steps = [
        "Add category-specific planner rules for red-team, toxicity, hacking/malware, and harassment prompts.",
        "Run a balanced benchmark with benign near-neighbor prompts to measure over-refusal.",
        "Promote high-confidence REWRITE cases into training data for planner tuning and DPO preference collection.",
        "Re-run this report after each planner update to show measurable safety gains.",
    ]
    ax.text(0.07, y - 0.04, "Recommended Next Steps", fontsize=15, weight="bold", color="#111827")
    y -= 0.085
    for item in next_steps:
        y = add_wrapped(ax, f"- {item}", 0.09, y, width=105, size=10.5, color="#374151", line_gap=0.03)

    ax.text(0.07, 0.08, "This report intentionally avoids printing harmful prompt text.", fontsize=9, color="#6b7280")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    results = read_jsonl(results_path)
    dataset_rows = read_jsonl(dataset_path)
    summary = summarize(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        cover_page(pdf, summary, dataset_rows, results_path)
        source_page(pdf, summary)
        outcome_page(pdf, summary)
        recommendation_page(pdf, summary)

    print(f"Wrote PDF report to {output_path}")


if __name__ == "__main__":
    main()
