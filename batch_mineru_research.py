#!/usr/bin/env python3
"""
MinerU 4.0.9 research batch runner - v13

Patches the actual formula recognizer classes directly and saves exact formula
crops during MFR. It also removes MinerU-labeled headers, footers, and page
numbers from the final Markdown, and conservatively filters noisy footnotes while preserving substantive scientific footnotes. The untouched MinerU Markdown is preserved as markdown_raw.md.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_ACTIVE_COLLECTOR = None
_PATCHED = []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class FormulaCollector:
    def __init__(self, paper_output: Path):
        self.paper_output = paper_output
        self.eq_dir = paper_output / "equations"
        self.eq_dir.mkdir(parents=True, exist_ok=True)
        self.page_offset = 0
        self.page_counts = {}
        self.items = []
        self.calls = []
        self.errors = []

    @staticmethod
    def rgb(image):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def bbox_pixels(bbox, width, height):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        x0, y0, x1, y1 = map(float, bbox)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.05:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        x0 = max(0, min(width, math.floor(x0)))
        y0 = max(0, min(height, math.floor(y0)))
        x1 = max(0, min(width, math.ceil(x1)))
        y1 = max(0, min(height, math.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def label(item):
        return str(item.get("label") or item.get("type") or "").strip().lower()

    @staticmethod
    def latex(item):
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""
        for key in ("latex", "math_content"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, dict):
            for key in ("latex", "math_content"):
                val = content.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return ""

    def capture(self, source_name, images_mfd_res, images, result):
        debug = {"source": source_name}
        try:
            if not isinstance(images, (list, tuple)):
                raise ValueError(f"images type={type(images).__name__}")
            if not isinstance(images_mfd_res, (list, tuple)):
                raise ValueError(f"detections type={type(images_mfd_res).__name__}")

            page_count = len(images)

            if len(images_mfd_res) == page_count and all(
                isinstance(x, (list, tuple)) for x in images_mfd_res
            ):
                det_pages = [list(x) for x in images_mfd_res]
            elif page_count == 1 and all(isinstance(x, dict) for x in images_mfd_res):
                det_pages = [list(images_mfd_res)]
            else:
                raise ValueError(
                    f"unexpected detection structure len={len(images_mfd_res)} pages={page_count}"
                )

            if (
                isinstance(result, (list, tuple))
                and len(result) == page_count
                and all(isinstance(x, (list, tuple)) for x in result)
            ):
                res_pages = [list(x) for x in result]
            else:
                res_pages = [[] for _ in range(page_count)]

            debug["page_count"] = page_count
            debug["detections_per_page"] = [len(x) for x in det_pages]
            saved = 0

            for local_page in range(page_count):
                doc_page = self.page_offset + local_page
                page_img = self.rgb(images[local_page])
                h, w = page_img.shape[:2]
                dets = det_pages[local_page]
                results = res_pages[local_page]

                for j, det in enumerate(dets):
                    if not isinstance(det, dict):
                        continue
                    lab = self.label(det)
                    if lab not in ("inline_formula", "display_formula"):
                        continue

                    box = self.bbox_pixels(det.get("bbox"), w, h)
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    crop = page_img[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue

                    kind = "display" if lab == "display_formula" else "inline"
                    latex = self.latex(det)
                    if j < len(results):
                        rlatex = self.latex(results[j])
                        if rlatex:
                            latex = rlatex

                    self.page_counts[doc_page] = self.page_counts.get(doc_page, 0) + 1
                    n = self.page_counts[doc_page]
                    filename = f"page_{doc_page+1:04d}_formula_{n:03d}_{kind}.png"
                    out = self.eq_dir / filename
                    Image.fromarray(crop).save(out)

                    self.items.append({
                        "page_idx": doc_page,
                        "page_number": doc_page + 1,
                        "formula_number_on_page": n,
                        "formula_type": kind,
                        "bbox_pixels": [x0, y0, x1, y1],
                        "bbox_normalized": [
                            round(x0 / w, 6), round(y0 / h, 6),
                            round(x1 / w, 6), round(y1 / h, 6)
                        ],
                        "latex": latex,
                        "image_path": str(out.relative_to(self.paper_output)).replace("\\", "/"),
                        "captured_from": source_name,
                    })
                    saved += 1

            self.page_offset += page_count
            debug["saved"] = saved

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            debug["error"] = msg
            self.errors.append(msg)

        self.calls.append(debug)

    def save(self):
        inter = self.paper_output / "intermediate"
        write_json(inter / "mfr_formula_manifest.json", {
            "formula_count": len(self.items),
            "capture_errors": self.errors,
            "formulas": self.items,
        })
        write_json(inter / "mfr_patch_debug.json", {
            "patched_methods": _PATCHED,
            "hook_calls": len(self.calls),
            "calls": self.calls,
            "capture_errors": self.errors,
        })


def patch_formula_recognizers():
    module_names = [
        "mineru.model.mfr.pp_formulanet.predict_formula",
        "mineru.model.mfr.pp_formulanet_plus_m_onnx",
        "mineru.model.mfr.unimernet.Unimernet",
        "mineru.model.mfr.pp_formulanet_plus_m.predict_formula",
    ]
    patched = []

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if getattr(cls, "__module__", None) != module.__name__:
                continue
            original = getattr(cls, "batch_predict", None)
            if not callable(original):
                continue

            try:
                source = inspect.getsource(original)
            except Exception:
                source = ""

            if "MFR Predict" not in source:
                continue

            source_name = f"{module.__name__}.{class_name}.batch_predict"

            def make_wrapper(orig, src_name):
                def wrapped(self, *args, **kwargs):
                    result = orig(self, *args, **kwargs)
                    collector = _ACTIVE_COLLECTOR
                    if collector is None:
                        return result

                    detections = (
                        args[0] if len(args) >= 1
                        else kwargs.get("images_mfd_res")
                        or kwargs.get("mfd_res")
                    )
                    images = (
                        args[1] if len(args) >= 2
                        else kwargs.get("images")
                        or kwargs.get("image")
                    )

                    if isinstance(images, np.ndarray):
                        images = [images]
                    if isinstance(detections, list) and detections and isinstance(detections[0], dict):
                        if isinstance(images, (list, tuple)) and len(images) == 1:
                            detections = [detections]

                    if detections is None or images is None:
                        collector.errors.append(
                            f"{src_name}: could not identify detections/images"
                        )
                    else:
                        collector.capture(src_name, detections, images, result)
                    return result
                return wrapped

            setattr(cls, "batch_predict", make_wrapper(original, source_name))
            patched.append(source_name)

    if not patched:
        raise RuntimeError("No FormulaRecognizer batch_predict() implementation was patched.")

    _PATCHED.extend(patched)
    return patched


# ---------------------------------------------------------------------------
# Markdown noise cleanup
# ---------------------------------------------------------------------------

DEFAULT_NOISE_TYPES = {
    "header",
    "footer",
    "page_header",
    "page_footer",
    "page_number",
    "number",
}


def _flatten_block_text(value: Any) -> str:
    """
    Extract visible text from one MinerU block without using OCR/regex guesses.
    """
    parts = []

    def walk(obj):
        if isinstance(obj, str):
            if obj.strip():
                parts.append(obj.strip())
            return

        if isinstance(obj, dict):
            # Prefer explicit text/content fields.
            if isinstance(obj.get("text"), str):
                text = obj["text"].strip()
                if text:
                    parts.append(text)

            content = obj.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    parts.append(text)
            elif isinstance(content, (dict, list)):
                walk(content)

            # Older/intermediate MinerU structures may use keys such as
            # paragraph_content, title_content, page_number_content, etc.
            for key, val in obj.items():
                if key in {"text", "content", "bbox", "lines", "spans"}:
                    continue
                if key.endswith("_content") and isinstance(val, (dict, list, str)):
                    walk(val)

            return

        if isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(value)

    # Preserve ordering while removing accidental duplicate fragments.
    cleaned = []
    for part in parts:
        if not cleaned or cleaned[-1] != part:
            cleaned.append(part)

    return " ".join(cleaned).strip()


def collect_noise_texts(json_path: Path, noise_types: set[str]) -> list[dict[str, Any]]:
    """
    Collect text ONLY from blocks MinerU explicitly labeled as noise.
    """
    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    found = []

    def walk(obj, page_idx=None):
        if isinstance(obj, dict):
            local_page = obj.get("page_idx", page_idx)
            block_type = str(
                obj.get("type")
                or obj.get("label")
                or obj.get("block_type")
                or ""
            ).strip().lower()

            if block_type in noise_types:
                text = _flatten_block_text(obj)
                if text:
                    found.append({
                        "type": block_type,
                        "page_idx": local_page,
                        "text": text,
                    })
                # Do not recursively treat nested text spans as separate blocks.
                return

            for key, val in obj.items():
                if key in {"bbox", "lines", "spans"}:
                    continue
                walk(val, local_page)

        elif isinstance(obj, list):
            for item in obj:
                walk(item, page_idx)

    walk(data)
    return found


def clean_markdown_from_labeled_noise(
    paper_output: Path,
    remove_aside_text: bool = False,
) -> dict[str, Any]:
    """
    Remove headers, footers and page numbers from markdown.md using MinerU's
    own block labels.

    The original MinerU Markdown is preserved as markdown_raw.md.
    """
    markdown_path = paper_output / "markdown.md"

    report = {
        "cleaned": False,
        "source": None,
        "noise_types": sorted(DEFAULT_NOISE_TYPES),
        "noise_blocks_found": 0,
        "lines_removed": 0,
        "removed": [],
        "warnings": [],
    }

    if not markdown_path.exists():
        report["warnings"].append("markdown.md was not found.")
        return report

    noise_types = set(DEFAULT_NOISE_TYPES)

    if remove_aside_text:
        noise_types.add("aside_text")
        noise_types.add("page_aside_text")

    report["noise_types"] = sorted(noise_types)

    # model_output.json is preferred because it is closest to the model labels.
    candidates = [
        paper_output / "model_output.json",
        paper_output / "structured_content.json",
        paper_output / "middle_json.json",
    ]

    noise_blocks = []

    for candidate in candidates:
        noise_blocks = collect_noise_texts(candidate, noise_types)
        if noise_blocks:
            report["source"] = candidate.name
            break

    if not noise_blocks:
        report["warnings"].append(
            "No labeled header/footer/page-number blocks were found."
        )
        return report

    report["noise_blocks_found"] = len(noise_blocks)

    raw_backup = paper_output / "markdown_raw.md"
    if not raw_backup.exists():
        shutil.copy2(markdown_path, raw_backup)

    original = markdown_path.read_text(encoding="utf-8", errors="replace")
    lines = original.splitlines()

    # Match conservatively: only exact Markdown lines that equal text from
    # MinerU-labeled noise blocks. We do NOT remove arbitrary repeated text.
    noise_counts: dict[str, int] = {}
    noise_meta: dict[str, list[dict[str, Any]]] = {}

    for item in noise_blocks:
        text = re.sub(r"\s+", " ", item["text"]).strip()
        if not text:
            continue

        # Headers/footers may contain multiple visible lines. Store the whole
        # block and each non-empty source line when possible.
        candidates_for_item = {text}

        for candidate_text in candidates_for_item:
            noise_counts[candidate_text] = noise_counts.get(candidate_text, 0) + 1
            noise_meta.setdefault(candidate_text, []).append(item)

    cleaned_lines = []
    removed = []

    for line_number, line in enumerate(lines, start=1):
        stripped = re.sub(r"\s+", " ", line.strip())

        # Do not strip Markdown heading markers. This helps preserve a genuine
        # "## Methods" heading even if a running header says "Methods".
        if stripped in noise_counts and noise_counts[stripped] > 0:
            meta = noise_meta[stripped][0] if noise_meta.get(stripped) else {}
            noise_counts[stripped] -= 1

            removed.append({
                "markdown_line": line_number,
                "text": line,
                "matched_type": meta.get("type"),
                "page_idx": meta.get("page_idx"),
            })
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    if original.endswith("\n"):
        cleaned += "\n"

    markdown_path.write_text(cleaned, encoding="utf-8")

    report["cleaned"] = True
    report["lines_removed"] = len(removed)
    report["removed"] = removed
    report["raw_backup"] = "markdown_raw.md"
    report["clean_markdown"] = "markdown.md"

    write_json(
        paper_output / "intermediate" / "markdown_cleanup_report.json",
        report,
    )

    return report



# ---------------------------------------------------------------------------
# Footnote cleanup
# ---------------------------------------------------------------------------

FOOTNOTE_TYPES = {
    "footnote",
    "page_footnote",
}

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"(?:\+?\d[\d\s().-]{7,}\d)"
)

NOISY_FOOTNOTE_PATTERNS = [
    ("correspondence_notice", re.compile(
        r"\b(?:corresponding author|correspondence should be addressed|"
        r"to whom correspondence|author to whom correspondence|"
        r"contact author)\b",
        flags=re.IGNORECASE,
    )),
    ("email_metadata", re.compile(
        r"\b(?:e-?mail|email address|electronic mail)\b",
        flags=re.IGNORECASE,
    )),
    ("present_address", re.compile(
        r"\b(?:present address|current address|permanent address|"
        r"author affiliation|affiliation)\b",
        flags=re.IGNORECASE,
    )),
    ("publisher_boilerplate", re.compile(
        r"\b(?:all rights reserved|published by|publisher|"
        r"under exclusive licen[cs]e|springer nature|"
        r"reprints? and permissions?|sharing guidelines|"
        r"terms and conditions|downloaded via)\b",
        flags=re.IGNORECASE,
    )),
    ("copyright_notice", re.compile(
        r"(?:©|\bcopyright\b|\(c\)\s*\d{4})",
        flags=re.IGNORECASE,
    )),
    ("orcid_metadata", re.compile(
        r"\bORCID\b",
        flags=re.IGNORECASE,
    )),
    ("phone_fax_metadata", re.compile(
        r"\b(?:tel(?:ephone)?|fax)\s*[:.]?",
        flags=re.IGNORECASE,
    )),
]

# Signals that a footnote carries scientific/technical meaning.
USEFUL_FOOTNOTE_PATTERNS = [
    ("alternative_definition", re.compile(
        r"\b(?:alternatively|equivalently|can be viewed as|"
        r"can also be understood as)\b",
        flags=re.IGNORECASE,
    )),
    ("definition_or_notation", re.compile(
        r"\b(?:we define|defined as|denotes?|refers to|means|"
        r"for conciseness|notation|by .* we mean)\b",
        flags=re.IGNORECASE,
    )),
    ("methodological_note", re.compile(
        r"\b(?:we also|we use|we used|we consider|we assume|"
        r"we experimented|we evaluate|we report|we selected|"
        r"in this work|in our experiments?|for our experiments?)\b",
        flags=re.IGNORECASE,
    )),
    ("qualification", re.compile(
        r"\b(?:note that|more precisely|specifically|in particular|"
        r"however|except|unless|respectively|for completeness)\b",
        flags=re.IGNORECASE,
    )),
    ("cross_reference", re.compile(
        r"\b(?:Fig\.?|Figure|Table|Sect\.?|Section|Eq\.?|Equation)\s*\d+",
        flags=re.IGNORECASE,
    )),
]

TECHNICAL_SYMBOL_RE = re.compile(
    r"(?:\\[A-Za-z]+|\$[^$]+\$|[=<>≤≥∈∉⊆⊂∀∃→↔]|"
    r"\b(?:iff|a\.k\.a\.|i\.e\.|e\.g\.)\b)",
    flags=re.IGNORECASE,
)

AFFILIATION_WORD_RE = re.compile(
    r"\b(?:department|university|institute|institution|laboratory|"
    r"school of|faculty of|college of|research center|research centre)\b",
    flags=re.IGNORECASE,
)


def collect_footnotes(json_path: Path) -> list[dict[str, Any]]:
    """
    Collect blocks MinerU explicitly labels as footnotes.
    """
    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    found = []

    def walk(obj, page_idx=None):
        if isinstance(obj, dict):
            local_page = obj.get("page_idx", page_idx)

            block_type = str(
                obj.get("type")
                or obj.get("label")
                or obj.get("block_type")
                or ""
            ).strip().lower()

            if block_type in FOOTNOTE_TYPES:
                text = _flatten_block_text(obj)

                if text:
                    found.append({
                        "type": block_type,
                        "page_idx": local_page,
                        "text": text,
                    })

                return

            for key, value in obj.items():
                if key in {"bbox", "lines", "spans"}:
                    continue
                walk(value, local_page)

        elif isinstance(obj, list):
            for item in obj:
                walk(item, page_idx)

    walk(data)
    return found


def classify_footnote(text: str) -> dict[str, Any]:
    """
    Conservative hybrid classifier.

    It removes only high-confidence administrative/contact/publisher noise.
    Scientifically meaningful or uncertain footnotes are kept, which minimizes
    the risk of deleting useful context for an LLM.
    """
    normalized = re.sub(r"\s+", " ", text).strip()

    noise_hits = []
    useful_hits = []

    if EMAIL_RE.search(normalized):
        noise_hits.append("email_address")

    if PHONE_RE.search(normalized):
        # Telephone-like patterns alone are weak evidence, so they only become
        # decisive when combined with other metadata indicators below.
        noise_hits.append("phone_number")

    for name, pattern in NOISY_FOOTNOTE_PATTERNS:
        if pattern.search(normalized):
            noise_hits.append(name)

    for name, pattern in USEFUL_FOOTNOTE_PATTERNS:
        if pattern.search(normalized):
            useful_hits.append(name)

    if TECHNICAL_SYMBOL_RE.search(normalized):
        useful_hits.append("technical_notation")

    affiliation_like = bool(AFFILIATION_WORD_RE.search(normalized))

    strong_noise = any(
        hit in {
            "correspondence_notice",
            "email_metadata",
            "publisher_boilerplate",
            "copyright_notice",
            "present_address",
            "orcid_metadata",
            "phone_fax_metadata",
        }
        for hit in noise_hits
    )

    # An actual email address is high-confidence noise when the footnote also
    # looks like author/contact metadata.
    contact_noise = (
        "email_address" in noise_hits
        and (
            strong_noise
            or affiliation_like
            or len(normalized) < 240
        )
    )

    # Short institutional lines are usually author-affiliation metadata.
    affiliation_noise = (
        affiliation_like
        and len(normalized) < 220
        and not useful_hits
    )

    # Preserve mixed/technical footnotes if they contain substantive signals.
    if useful_hits:
        decision = "keep"
        reason = "substantive_or_technical_content"

    elif strong_noise or contact_noise or affiliation_noise:
        decision = "remove"
        reason = "administrative_or_contact_noise"

    else:
        # Ambiguous footnotes are kept intentionally. False-positive deletion
        # is more damaging for RAG than retaining a small amount of noise.
        decision = "keep"
        reason = "uncertain_kept_conservatively"

    return {
        "decision": decision,
        "reason": reason,
        "noise_signals": sorted(set(noise_hits)),
        "useful_signals": sorted(set(useful_hits)),
        "affiliation_like": affiliation_like,
    }


def _normalize_markdown_block_for_match(text: str) -> str:
    """
    Normalize one Markdown paragraph/block for comparison with a footnote.
    """
    text = text.strip()

    # Remove common Markdown footnote/list/superscript-style prefixes.
    text = re.sub(
        r"^(?:\[\^?\d+\]:?|\[\d+\]|\(?\d+\)?[.)]?|<sup>\d+</sup>)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove simple emphasis markers without changing content.
    text = text.replace("**", "").replace("__", "")

    return re.sub(r"\s+", " ", text).strip()



def _normalized_tokens(text: str) -> list[str]:
    text = _normalize_markdown_block_for_match(text).lower()
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", text)


def _extract_doi(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;)") if match else None


def _footnote_match_score(markdown_block: str, footnote_text: str) -> float:
    """
    Conservative similarity score used only for footnotes already classified
    as high-confidence noise.
    """
    a = _normalize_markdown_block_for_match(markdown_block)
    b = _normalize_markdown_block_for_match(footnote_text)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # DOI equality is a very strong identifier for publication boilerplate.
    doi = _extract_doi(b)
    if doi and doi.lower() in a.lower():
        return 0.995

    # Exact containment after normalization.
    if len(b) >= 20 and b in a:
        return 0.98
    if len(a) >= 20 and a in b:
        return 0.96

    seq = SequenceMatcher(None, a, b).ratio()

    ta = set(_normalized_tokens(a))
    tb = set(_normalized_tokens(b))
    if ta and tb:
        # How much of the shorter text is represented in the longer one.
        overlap = len(ta & tb) / min(len(ta), len(tb))
    else:
        overlap = 0.0

    # Long shared phrase check. This handles MinerU splitting one footnote into
    # multiple Markdown fragments.
    words_b = _normalized_tokens(b)
    phrase_hit = 0.0
    if len(words_b) >= 6:
        lower_a = a.lower()
        for n in (10, 8, 6):
            if len(words_b) < n:
                continue
            for i in range(len(words_b) - n + 1):
                phrase = " ".join(words_b[i:i+n])
                if phrase in lower_a:
                    phrase_hit = max(phrase_hit, 0.97 if n >= 8 else 0.92)
                    break
            if phrase_hit:
                break

    return max(seq, overlap, phrase_hit)


def _remove_noisy_fragment_from_block(block: str, footnote_text: str) -> tuple[str, bool, float]:
    """
    If a noisy footnote has been merged into a larger Markdown block, remove
    only the matching line/sentence-like fragment rather than the whole block.
    """
    target = _normalize_markdown_block_for_match(footnote_text)
    if not target:
        return block, False, 0.0

    # Try line-by-line first.
    lines = block.splitlines()
    best_i = None
    best_score = 0.0

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        score = _footnote_match_score(line, target)
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= 0.68:
        del lines[best_i]
        return "\n".join(lines), True, best_score

    # Try sentence-like fragments for cases where the noise is embedded in a
    # paragraph. Preserve the rest of the paragraph.
    parts = re.split(r"(?<=[.!?])\s+", block)
    best_i = None
    best_score = 0.0

    for i, part in enumerate(parts):
        if not part.strip():
            continue
        score = _footnote_match_score(part, target)
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= 0.68:
        del parts[best_i]
        return " ".join(parts), True, best_score

    return block, False, best_score


def remove_noisy_footnotes_from_markdown(
    paper_output: Path,
) -> dict[str, Any]:
    """
    Classify MinerU-labeled footnotes and remove only high-confidence noisy
    footnotes from markdown.md.

    v13 handles three cases:
      1. whole Markdown block matches the noisy footnote;
      2. the noisy footnote is split across a line in a larger block;
      3. the noisy footnote is embedded as a sentence-like fragment.

    markdown_raw.md remains untouched.
    """
    markdown_path = paper_output / "markdown.md"

    report = {
        "source": None,
        "footnotes_found": 0,
        "kept": 0,
        "removed": 0,
        "matched_and_removed_from_markdown": 0,
        "items": [],
        "warnings": [],
    }

    if not markdown_path.exists():
        report["warnings"].append("markdown.md was not found.")
        return report

    candidates = [
        paper_output / "model_output.json",
        paper_output / "structured_content.json",
        paper_output / "middle_json.json",
    ]

    footnotes = []

    for candidate in candidates:
        footnotes = collect_footnotes(candidate)
        if footnotes:
            report["source"] = candidate.name
            break

    if not footnotes:
        report["warnings"].append(
            "No MinerU-labeled footnote blocks were found."
        )
        write_json(
            paper_output / "intermediate" / "footnote_cleanup_report.json",
            report,
        )
        return report

    report["footnotes_found"] = len(footnotes)

    noisy_items = []

    for item in footnotes:
        classification = classify_footnote(item["text"])

        record = {
            "page_idx": item.get("page_idx"),
            "type": item.get("type"),
            "text": item["text"],
            **classification,
        }

        report["items"].append(record)

        if classification["decision"] == "remove":
            report["removed"] += 1
            noisy_items.append(record)
        else:
            report["kept"] += 1

    original = markdown_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    blocks = re.split(r"(\n\s*\n)", original)

    removed_from_markdown = 0

    for noisy in noisy_items:
        # First look for a whole-block match.
        best_idx = None
        best_score = 0.0

        for idx, block in enumerate(blocks):
            if not block or re.fullmatch(r"\n\s*\n", block):
                continue

            score = _footnote_match_score(block, noisy["text"])
            if score > best_score:
                best_score = score
                best_idx = idx

        # Whole-block removal threshold stays conservative.
        if best_idx is not None and best_score >= 0.68:
            blocks[best_idx] = ""
            removed_from_markdown += 1
            noisy["markdown_match_score"] = round(best_score, 4)
            noisy["removed_from_markdown"] = True
            noisy["removal_mode"] = "whole_block"
            continue

        # If the whole block was not strong enough, try removing only the
        # matching fragment from each block.
        fragment_removed = False
        fragment_best = best_score

        for idx, block in enumerate(blocks):
            if not block or re.fullmatch(r"\n\s*\n", block):
                continue

            new_block, changed, score = _remove_noisy_fragment_from_block(
                block,
                noisy["text"],
            )

            fragment_best = max(fragment_best, score)

            if changed:
                blocks[idx] = new_block
                removed_from_markdown += 1
                noisy["markdown_match_score"] = round(score, 4)
                noisy["removed_from_markdown"] = True
                noisy["removal_mode"] = "fragment"
                fragment_removed = True
                break

        if not fragment_removed:
            noisy["markdown_match_score"] = round(fragment_best, 4)
            noisy["removed_from_markdown"] = False
            noisy["removal_mode"] = None

    cleaned = "".join(blocks)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    markdown_path.write_text(
        cleaned,
        encoding="utf-8",
    )

    report["matched_and_removed_from_markdown"] = removed_from_markdown

    unmatched = report["removed"] - removed_from_markdown
    if unmatched:
        report["warnings"].append(
            f"{unmatched} noisy footnote block(s) were classified for removal "
            "but could not be matched safely in markdown.md; they were left unchanged."
        )

    write_json(
        paper_output / "intermediate" / "footnote_cleanup_report.json",
        report,
    )

    return report


def create_layout_pdf(source_pdf, middle_path, output_pdf):
    try:
        from docvortex.visualization import render_layout_pdf
        from mineru.types import MiddleJson
        middle = MiddleJson.from_json(middle_path.read_text(encoding="utf-8"))
        output_pdf.write_bytes(
            render_layout_pdf(source_pdf.read_bytes(), middle.pages)
        )
        return True, "docvortex.visualization.render_layout_pdf"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tier", default="standard",
                        choices=["flash", "basic", "standard", "advanced"])
    parser.add_argument("--ocr-mode", default="auto",
                        choices=["auto", "on", "off"])
    parser.add_argument(
        "--remove-aside-text",
        action="store_true",
        help="Also remove MinerU-labeled aside/margin text from final Markdown.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    pdfs = sorted(p for p in input_dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        raise SystemExit("No PDFs found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    patched = patch_formula_recognizers()

    from mineru.parser import parse
    from mineru.parser.writer import FileBasedDataWriter

    print("Patched MFR method(s):")
    for p in patched:
        print(" ", p)
    print()

    global _ACTIVE_COLLECTOR

    for i, pdf in enumerate(pdfs, 1):
        paper_out = output_dir / pdf.stem
        paper_out.mkdir(parents=True, exist_ok=True)
        collector = FormulaCollector(paper_out)

        print(f"[{i}/{len(pdfs)}] {pdf.name}")

        _ACTIVE_COLLECTOR = collector
        result = parse(pdf, tier=args.tier, ocr_mode=args.ocr_mode, page_range="all")
        _ACTIVE_COLLECTOR = None

        result.save(FileBasedDataWriter(str(paper_out)))
        collector.save()

        cleanup_report = clean_markdown_from_labeled_noise(
            paper_out,
            remove_aside_text=args.remove_aside_text,
        )

        footnote_report = remove_noisy_footnotes_from_markdown(
            paper_out,
        )

        middle = paper_out / "middle_json.json"
        layout_ok = False
        if middle.exists():
            layout_ok, _ = create_layout_pdf(
                pdf, middle, paper_out / f"{pdf.stem}_layout.pdf"
            )

        print(f"    layout PDF: {'yes' if layout_ok else 'NO'}")
        print(f"    recognizer hook calls: {len(collector.calls)}")
        print(f"    equation images: {len(collector.items)}")
        print(f"    Markdown noise removed: {cleanup_report.get('lines_removed', 0)} line(s)")
        print(
            "    footnotes: "
            f"{footnote_report.get('kept', 0)} kept, "
            f"{footnote_report.get('removed', 0)} classified as noise, "
            f"{footnote_report.get('matched_and_removed_from_markdown', 0)} removed from Markdown"
        )
        print("    raw Markdown backup: markdown_raw.md")

        for err in collector.errors:
            print("    CAPTURE WARNING:", err)


if __name__ == "__main__":
    main()
