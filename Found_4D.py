from __future__ import annotations

from pathlib import Path
import re


ANALYSIS_PATH = Path("Analysis_4D_27_Jan_2026.md")
PREDICT_PATH = Path("Predict_4D_26_Jan_2026.md")
OUTPUT_PATH = Path("Found_4D_26_Jan_2026.md")


def extract_analysis_sections(text: str) -> dict[str, list[str]]:
    sections_for_number: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_section = stripped[4:].strip()
            continue

        if not current_section:
            continue

        for match in re.findall(r"`(\d{4})`", line):
            key = normalize_digits(match)
            sections = sections_for_number.setdefault(key, [])
            if not sections or sections[-1] != current_section:
                if current_section not in sections:
                    sections.append(current_section)

    return sections_for_number


def extract_predict_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| Number ") and "Model Probability" in stripped:
            in_table = True
            continue
        if in_table:
            if not stripped.startswith("|"):
                if not stripped or stripped.startswith("##"):
                    break
                continue
            # Skip the header separator row.
            if set(stripped.replace("|", "").strip()) <= set(":- "):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells:
                continue
            first = cells[0]
            if re.fullmatch(r"\d{4}", first):
                numbers.append(first)
    return numbers


def normalize_digits(value: str) -> str:
    return "".join(sorted(value))


def main() -> None:
    analysis_text = ANALYSIS_PATH.read_text(encoding="utf-8")
    predict_text = PREDICT_PATH.read_text(encoding="utf-8")

    analysis_sections = extract_analysis_sections(analysis_text)

    predict_numbers = extract_predict_numbers(predict_text)
    found: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    for num in predict_numbers:
        if num in seen:
            continue
        key = normalize_digits(num)
        sections = analysis_sections.get(key)
        if sections:
            found.append((num, sections))
            seen.add(num)

    lines = [
        "# Found 4D Numbers for 18 Jan 2026",
        "",
        f"- Analysis source: `{ANALYSIS_PATH.name}`",
        f"- Prediction source: `{PREDICT_PATH.name}`",
        f"- Total predicted numbers checked: {len(predict_numbers)}",
        f"- Found (including digit permutations): {len(found)}",
        "",
    ]

    if not found:
        lines.append("No predicted numbers matched the analysis combinations.")
    else:
        lines.append("| Number | Found In |")
        lines.append("|---|---|")
        lines.extend(
            f"| {num} | {'; '.join(sections)} |" for num, sections in found
        )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
