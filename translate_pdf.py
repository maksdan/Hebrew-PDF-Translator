#!/usr/bin/env python3
"""
Hebrew PDF Translator
Extracts text from Hebrew PDFs and translates to English using Claude.

Usage:
  python translate_pdf.py document.pdf
  python translate_pdf.py document.pdf --pages 1-10
  python translate_pdf.py document.pdf --pages 1,3,5-8
  python translate_pdf.py document.pdf --no-translate
  python translate_pdf.py ./pdfs_folder/          # process all PDFs in a directory
  python translate_pdf.py a.pdf b.pdf c.pdf       # multiple files
"""

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

# Unicode bidi/directional control characters inserted by pdftotext
_BIDI_CHARS = set("‎‏‪‫‬‭‮⁦⁧⁨⁩")


def extract_text(pdf_path: str, pages: list[int] | None = None) -> dict[int, str]:
    """Return {1-indexed page num: text} via pdftotext (Poppler).

    Applies the same Unicode font mapping and bidi reordering that PDF readers
    use on copy-paste, giving clean correctly-ordered Hebrew text.
    """
    import subprocess

    cmd = ["pdftotext"]
    if pages is not None:
        cmd += ["-f", str(min(pages) + 1), "-l", str(max(pages) + 1)]
    cmd += [pdf_path, "-"]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except FileNotFoundError:
        print("Error: pdftotext not found. Install with: brew install poppler")
        sys.exit(1)

    if proc.returncode != 0:
        print(f"Error: pdftotext failed: {proc.stderr.decode()}")
        sys.exit(1)

    raw = proc.stdout.decode("utf-8")
    clean = "".join(c for c in raw if c not in _BIDI_CHARS)

    page_chunks = clean.split("\f")  # pdftotext separates pages with form-feed

    result: dict[int, str] = {}
    if pages is not None:
        first = min(pages)
        for i, chunk in enumerate(page_chunks):
            page_0idx = first + i
            if page_0idx in pages:
                result[page_0idx + 1] = chunk.strip()
    else:
        for i, chunk in enumerate(page_chunks):
            if chunk.strip():
                result[i + 1] = chunk.strip()

    return result


_RASHI_MODEL_URL = (
    "https://gitlab.com/pninim.org/tessdata_heb_rashi/-/raw/main/"
    "tesseract_4.1.1/heb_rashi.traineddata"
)
_TESSDATA_CANDIDATES = [
    Path("/opt/homebrew/share/tessdata"),   # Apple Silicon Mac (Homebrew)
    Path("/usr/local/share/tessdata"),       # Intel Mac / Linux Homebrew
    Path("/usr/share/tesseract-ocr/4.00/tessdata"),  # Ubuntu/Debian
    Path("/usr/share/tessdata"),
]


def find_tessdata_dir() -> Path | None:
    for p in _TESSDATA_CANDIDATES:
        if p.is_dir():
            return p
    return None


def download_rashi_model() -> bool:
    """Download heb_rashi.traineddata into the Tesseract tessdata directory."""
    import urllib.request

    tessdata = find_tessdata_dir()
    if tessdata is None:
        print("Error: could not locate Tesseract tessdata directory.")
        print("Searched:", [str(p) for p in _TESSDATA_CANDIDATES])
        return False

    dest = tessdata / "heb_rashi.traineddata"
    if dest.exists():
        print(f"Rashi model already installed: {dest}")
        return True

    print(f"Downloading Rashi model → {dest} ...")
    try:
        urllib.request.urlretrieve(_RASHI_MODEL_URL, dest)
        print("Done.")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        print(f"You can manually download from:\n  {_RASHI_MODEL_URL}")
        print(f"and place it at:\n  {dest}")
        return False


def extract_text_ocr(pdf_path: str, pages: list[int] | None = None) -> dict[int, str]:
    """Return {1-indexed page num: text} using Tesseract OCR.

    Renders each page to a 300 DPI image and runs Hebrew OCR. Uses the
    heb_rashi model (Pninim project) when installed — run with --setup-rashi
    first to download it. Falls back to heb_old then heb.
    """
    try:
        import fitz  # PyMuPDF — for rendering pages to images
    except ImportError:
        print("Error: pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("Error: pytesseract/Pillow not installed. Run: pip install pytesseract Pillow")
        sys.exit(1)

    available = pytesseract.get_languages()
    if "heb_rashi" in available:
        lang = "heb_rashi+heb"
        print("  Using heb_rashi model (Rashi-script aware)")
    elif "heb_old" in available:
        lang = "heb+heb_old"
        print("  Using heb+heb_old (tip: run --setup-rashi for better Rashi results)")
    else:
        lang = "heb"
        print("  Using heb only (tip: run --setup-rashi for better Rashi results)")

    doc = fitz.open(pdf_path)
    total = len(doc)
    result: dict[int, str] = {}

    indices = pages if pages is not None else list(range(total))
    for idx in indices:
        if idx < 0 or idx >= total:
            print(f"  Warning: page {idx + 1} skipped (PDF has {total} pages)")
            continue

        # Render at 300 DPI for good OCR accuracy
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # psm 3 = fully automatic page segmentation; oem 1 = LSTM engine
        config = "--psm 3 --oem 1"
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        result[idx + 1] = text.strip()

    doc.close()
    return result


def parse_pages(spec: str) -> list[int]:
    """Parse '1-5', '1,3,5', or '1-3,7,9-11' into a sorted 0-indexed list."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            indices.update(range(int(a) - 1, int(b)))
        else:
            indices.add(int(part) - 1)
    return sorted(indices)


# ---------------------------------------------------------------------------
# Hebrew cleanup (OCR de-glitch)
# ---------------------------------------------------------------------------

def clean_hebrew_with_claude(texts: dict[int, str], model: str) -> dict[int, str]:
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    cleaned: dict[int, str] = {}

    system = [
        {
            "type": "text",
            "text": (
                "You are an expert in classical and rabbinic Hebrew texts, including Rashi semi-cursive script. "
                "The text you receive was produced by OCR on scanned historical Hebrew books and contains errors. "
                "Your job is to produce correct, coherent Hebrew — this means two things:\n\n"
                "1. Fix OCR artifacts: remove stray characters, fix garbled words, correct obvious misreads.\n\n"
                "2. Reconstruct ambiguous letters using meaning: Rashi script has letter-pairs that look nearly "
                "identical (e.g. ר/ד, ו/ז, ה/ח/ת, נ/ג, י/ו, כ/בּ). When the OCR picks the wrong one, the word "
                "may be nonsensical or grammatically broken. Actively evaluate each suspicious word in context — "
                "ask which letter choice produces coherent Hebrew given the surrounding grammar, syntax, and subject "
                "matter — and substitute accordingly. Do not leave a word as-is simply because the OCR produced "
                "something letter-shaped; if it doesn't make sense, fix it.\n\n"
                "Use your knowledge of biblical, Talmudic, and rabbinic literature to judge what the text is "
                "trying to say. When confident, correct. When uncertain between two readings, pick the one that "
                "fits the context best.\n\n"
                "Rules: output only the corrected Hebrew — no translation, no transliteration, no commentary. "
                "Preserve paragraph structure and line breaks. "
                "If a passage is too damaged to recover, output your best reconstruction rather than omitting it."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    for page_num, text in sorted(texts.items()):
        if not text.strip():
            cleaned[page_num] = ""
            continue

        print(f"  Page {page_num}...", end=" ", flush=True)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            cleaned[page_num] = response.content[0].text
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc} — keeping raw OCR text")
            cleaned[page_num] = text  # fall back to raw rather than losing the page

        time.sleep(0.4)

    return cleaned


# ---------------------------------------------------------------------------
# Rashi reading validator
# ---------------------------------------------------------------------------

def validate_rashi_reading(sentence: str, model: str) -> None:
    """Check whether a Hebrew sentence transcribed from Rashi script is coherent."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    system = (
        "You are an expert in classical and rabbinic Hebrew, specializing in Rashi semi-cursive script. "
        "A scholar is hand-reading a scanned manuscript written in Rashi script and has typed what they "
        "believe the text says. Because certain letters are visually similar in Rashi script, "
        "they may have misread one or two characters in a word.\n\n"
        "The most common Rashi script confusable pairs are:\n"
        "  • ד ↔ ר  (dalet / resh)\n"
        "  • ו ↔ ז  (vav / zayin)\n"
        "  • ה ↔ ח ↔ ת  (he / chet / tav)\n"
        "  • י ↔ ו  (yod / vav)\n"
        "  • כ ↔ ב  (kaf / bet)\n"
        "  • נ ↔ ג  (nun / gimel)\n"
        "  • מ ↔ ס  (mem / samech)\n\n"
        "Your task:\n"
        "1. Read the submitted sentence as Hebrew (square or rabbinic — the scholar has already "
        "transliterated it into Unicode Hebrew for you).\n"
        "2. Determine whether the sentence is coherent biblical, Talmudic, or rabbinic Hebrew.\n"
        "3. Flag any word that looks wrong due to a likely character misread (applying the confusable "
        "pairs above). For each flagged word, state the suspicious word, the most plausible corrected "
        "word, and which letter swap explains it.\n"
        "4. If corrections are needed, provide the most likely intended reading of the full sentence.\n\n"
        "Output format — use exactly these four sections, even if some are empty:\n\n"
        "VERDICT: [Coherent | Likely error | Unclear]\n\n"
        "FLAGGED WORDS:\n"
        "<word as submitted> → <corrected word>  (swap: X ↔ Y)\n"
        "(or 'None' if the sentence is fully coherent)\n\n"
        "LIKELY INTENDED TEXT:\n"
        "<corrected full sentence, or 'Same as submitted' if no changes>\n\n"
        "NOTES:\n"
        "<brief explanation of your reasoning, or anything else the scholar should know>"
    )

    print("Validating...")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": sentence}],
        )
        print()
        print(response.content[0].text)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_with_claude(texts: dict[int, str], model: str) -> dict[int, str]:
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    translations: dict[int, str] = {}

    # Cache the system prompt across all pages to reduce cost on large docs.
    system = [
        {
            "type": "text",
            "text": (
                "You are an expert translator of classical and rabbinic Hebrew texts. "
                "The text you receive is extracted via OCR from historical Hebrew printed books "
                "and may contain minor OCR errors (misread letters, missing vowels, stray characters). "
                "Use your knowledge of biblical, Talmudic, and rabbinic literature to infer the "
                "correct reading when the OCR output is imperfect. "
                "The text may include a mix of: square Hebrew script, Rashi semi-cursive script, "
                "Aramaic (Targum Onkelos or other Targumim), and occasional Persian or other languages. "
                "Translate everything to natural, accurate English. "
                "Preserve paragraph structure and line breaks. "
                "For Aramaic passages, translate them as you would Hebrew — do not transliterate. "
                "Output only the translation — no commentary, no headers, no notes."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    for page_num, text in sorted(texts.items()):
        if not text.strip():
            translations[page_num] = ""
            continue

        print(f"  Page {page_num}...", end=" ", flush=True)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            translations[page_num] = response.content[0].text
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            translations[page_num] = f"[Translation error: {exc}]"

        time.sleep(0.4)  # stay inside rate limits

    return translations


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(page_texts: dict[int, str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for page_num, text in sorted(page_texts.items()):
            fh.write(f"{'=' * 60}\n")
            fh.write(f"Page {page_num}\n")
            fh.write(f"{'=' * 60}\n")
            fh.write(text.strip())
            fh.write("\n\n")
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    pages: list[int] | None,
    extractor: str,
    model: str,
    no_translate: bool,
    no_clean: bool,
) -> None:
    stem = pdf_path.stem
    hebrew_out = output_dir / f"{stem}_hebrew.txt"
    hebrew_clean_out = output_dir / f"{stem}_hebrew_clean.txt"
    english_out = output_dir / f"{stem}_english.txt"

    print(f"\n{'─' * 60}")
    print(f"PDF: {pdf_path}")
    if pages:
        print(f"  Pages filter: {[p + 1 for p in pages]}")
    print(f"  Extractor: {extractor}")

    if extractor == "ocr":
        texts = extract_text_ocr(str(pdf_path), pages)
    else:
        texts = extract_text(str(pdf_path), pages)

    if not texts:
        print("  No text extracted — skipping.")
        return
    print(f"  Extracted {len(texts)} page(s)")

    # For OCR output, run a Claude cleanup pass to fix misread characters
    # before translation. Skipped for pdftotext (already clean) or --no-clean.
    texts_for_translation = texts
    if extractor == "ocr" and not no_clean and not no_translate:
        print(f"Cleaning Hebrew OCR output with Claude ({model})...")
        texts_for_translation = clean_hebrew_with_claude(texts, model=model)
        write_output(texts_for_translation, hebrew_clean_out)
    else:
        write_output(texts, hebrew_out)

    if no_translate:
        return

    print(f"Translating with Claude ({model})...")
    translations = translate_with_claude(texts_for_translation, model=model)

    write_output(translations, english_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and translate Hebrew PDFs to English.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="PDF_OR_DIR",
        help="One or more PDF files, or a directory containing PDFs.",
    )
    parser.add_argument(
        "--pages", "-p",
        help=(
            "Pages to process, e.g. '1-5', '1,3,5', '1-3,7'. "
            "Only applies when a single PDF is given."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        help=(
            "Claude model ID. "
            "Options: claude-sonnet-4-6 (default), claude-opus-4-7 (higher quality), "
            "claude-haiku-4-5-20251001 (fastest/cheapest)."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Where to write output files. Default: alongside each PDF.",
    )
    parser.add_argument(
        "--extractor", "-e",
        choices=["pdftotext", "ocr"],
        default="pdftotext",
        help=(
            "Text extraction method. 'pdftotext' (default) reads embedded text directly. "
            "'ocr' renders each page as an image and runs Tesseract — handles Rashi script "
            "and fixes font-encoding problems at the cost of speed."
        ),
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Only extract Hebrew text; skip translation.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Skip the Hebrew OCR cleanup pass (only relevant with --extractor ocr). "
            "Translation will use raw OCR output instead of the de-glitched Hebrew."
        ),
    )
    parser.add_argument(
        "--setup-rashi",
        action="store_true",
        help=(
            "Download and install the Rashi-script Tesseract model "
            "(heb_rashi.traineddata from the Pninim project), then exit. "
            "Run this once before using --extractor ocr on Rashi-script PDFs."
        ),
    )
    parser.add_argument(
        "--validate", "-v",
        metavar="TEXT",
        help=(
            "Validate a Hebrew sentence you transcribed from a Rashi-script manuscript. "
            "Returns a verdict (coherent / likely error), flags suspicious words with "
            "their probable correct readings, and shows the likely intended text. "
            "No PDF input is needed when using this flag."
        ),
    )

    args = parser.parse_args()

    if args.validate:
        validate_rashi_reading(args.validate, model=args.model)
        sys.exit(0)

    if args.setup_rashi:
        success = download_rashi_model()
        sys.exit(0 if success else 1)

    if not args.inputs:
        parser.error("the following arguments are required: PDF_OR_DIR")

    # Resolve input files
    pdf_files: list[Path] = []
    for raw in args.inputs:
        p = Path(raw).resolve()
        if p.is_dir():
            found = sorted(p.glob("*.pdf")) + sorted(p.glob("*.PDF"))
            if not found:
                print(f"Warning: no PDFs found in {p}")
            pdf_files.extend(found)
        elif p.suffix.lower() == ".pdf":
            if not p.exists():
                print(f"Error: file not found: {p}")
                sys.exit(1)
            pdf_files.append(p)
        else:
            print(f"Warning: skipping non-PDF input: {p}")

    if not pdf_files:
        print("Error: no PDF files to process.")
        sys.exit(1)

    # --pages only makes sense for a single file
    pages: list[int] | None = None
    if args.pages:
        if len(pdf_files) > 1:
            print("Warning: --pages is ignored when processing multiple PDFs.")
        else:
            pages = parse_pages(args.pages)

    # Output directory
    shared_output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if shared_output_dir:
        shared_output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        out_dir = shared_output_dir if shared_output_dir else pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        process_pdf(
            pdf_path=pdf_path,
            output_dir=out_dir,
            pages=pages if len(pdf_files) == 1 else None,
            extractor=args.extractor,
            model=args.model,
            no_translate=args.no_translate,
            no_clean=args.no_clean,
        )

    print(f"\nAll done. Processed {len(pdf_files)} file(s).")


if __name__ == "__main__":
    main()
