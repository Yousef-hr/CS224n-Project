#!/usr/bin/env python3
"""
Download a subset of WikiText (train and test) and save as wiki-train.json and wiki-test.json
in cs224n-jepa-calm/. Output is JSONL: one JSON object per line with a "text" key.

By default produces document-level data: lines are grouped by section. A new document
starts at each level-1 header ( = Title = ) or level-2 header ( = = Section = = ), so each
output document is one article or one section of an article. Use --article-level for one
document per full article only; use --line-level for one row per line.
"""

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional

TEXT_KEY = "text"


def _is_level1_header(line: str) -> bool:
    """Level-1 only: '= Title =' (robust to leading spaces)."""
    s = line.strip()
    return bool(s) and s.startswith("= ") and not s.startswith("= = ")


def _is_section_header(line: str) -> bool:
    """Level-1/2/3 section header, robust to leading spaces."""
    s = line.strip()
    if not s:
        return False
    return s.startswith("= ")


def lines_to_documents(rows: Iterator[dict], section_level: bool = True) -> Iterator[str]:
    """Group WikiText lines; yield one concatenated document per article or per section."""
    is_start = _is_section_header if section_level else _is_level1_header
    buffer: list[str] = []

    def flush_buffer() -> Optional[str]:
        if not buffer:
            return None
        # Skip shards that contain only headings and no body text.
        has_body = any(not _is_section_header(line) for line in buffer)
        if not has_body:
            return None
        return "\n".join(buffer).strip()

    for row in rows:
        line = row.get(TEXT_KEY, "")
        stripped = line.strip()
        # New document starts only at header boundaries (not every blank line).
        if is_start(line) and buffer:
            doc = flush_buffer()
            if doc:
                yield doc
            buffer = []
        if stripped:
            buffer.append(line.rstrip("\n"))
    if buffer:
        doc = flush_buffer()
        if doc:
            yield doc


def main():
    parser = argparse.ArgumentParser(description="Download WikiText subset to cs224n-jepa-calm/")
    parser.add_argument(
        "--config",
        type=str,
        default="wikitext-103-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="WikiText config: 2 (small) or 103 (larger)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: project_root/cs224n-jepa-calm)",
    )
    parser.add_argument(
        "--article-level",
        action="store_true",
        help="One document per article only (level-1 header). Default is section-level (more, shorter docs).",
    )
    parser.add_argument(
        "--line-level",
        action="store_true",
        help="One JSONL row per line (no grouping).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Max number of train examples to keep (default: all)",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Max number of test examples to keep (default: all)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    out_dir = args.out_dir
    if out_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        out_dir = project_root / "cs224n-jepa-calm"
    else:
        out_dir = Path(out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WikiText config: {args.config}")
    ds = load_dataset("wikitext", args.config)

    def write_split_line_level(data, max_examples: Optional[int], out_path: Path) -> None:
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in data:
                if max_examples is not None and n >= max_examples:
                    break
                text = row.get(TEXT_KEY, "")
                if not text.strip():
                    continue
                f.write(json.dumps({TEXT_KEY: text}, ensure_ascii=False) + "\n")
                n += 1
        print(f"Wrote {n} line-level examples to {out_path}")

    def write_split_document_level(
        data, max_examples: Optional[int], out_path: Path, section_level: bool = True
    ) -> None:
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in lines_to_documents(iter(data), section_level=section_level):
                if max_examples is not None and n >= max_examples:
                    break
                f.write(json.dumps({TEXT_KEY: doc}, ensure_ascii=False) + "\n")
                n += 1
        print(f"Wrote {n} document-level examples to {out_path}")

    if args.line_level:
        write_split = write_split_line_level
    else:
        section_level = not args.article_level
        write_split = lambda data, max_ex, path: write_split_document_level(
            data, max_ex, path, section_level=section_level
        )
    train_data = ds["train"]
    test_data = ds["test"]
    write_split(train_data, args.max_train, out_dir / "wiki-train.json")
    write_split(test_data, args.max_test, out_dir / "wiki-test.json")


if __name__ == "__main__":
    main()
