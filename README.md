# Hebrew PDF Translator

Extract text from Hebrew PDFs and translate to English using Claude. Designed for classical and rabbinic Hebrew texts — biblical, Talmudic, and Rashi script.

## Features

- **Two extraction methods:** `pdftotext` for PDFs with embedded text, Tesseract OCR for scanned documents and Rashi script
- **Claude-powered translation** with a system prompt tuned for rabbinic Hebrew, Aramaic (Targumim), and OCR error correction
- **Batch processing:** single files, multiple files, or entire directories
- **Page selection:** process specific pages with `--pages 1-5,7,9-11`

## Prerequisites

- Python 3.10+
- [Poppler](https://poppler.freedesktop.org/) (provides `pdftotext`):
  ```bash
  brew install poppler        # macOS
  sudo apt install poppler-utils  # Ubuntu/Debian
  ```
- [Tesseract](https://github.com/tesseract-ocr/tesseract) (only needed for `--extractor ocr`):
  ```bash
  brew install tesseract tesseract-lang  # macOS
  sudo apt install tesseract-ocr tesseract-ocr-heb  # Ubuntu/Debian
  ```
- An [Anthropic API key](https://console.anthropic.com/) set as `ANTHROPIC_API_KEY`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Translate a single PDF
python translate_pdf.py document.pdf

# Only specific pages
python translate_pdf.py document.pdf --pages 1-10

# Extract Hebrew text without translating
python translate_pdf.py document.pdf --no-translate

# Use OCR for scanned PDFs or Rashi script
python translate_pdf.py document.pdf --extractor ocr

# Process all PDFs in a directory
python translate_pdf.py ./pdfs_folder/

# Multiple files
python translate_pdf.py a.pdf b.pdf c.pdf

# Use a different Claude model
python translate_pdf.py document.pdf --model claude-haiku-4-5-20251001
```

### Rashi Script Support

For better OCR on Rashi semi-cursive script, install the dedicated Tesseract model:

```bash
python translate_pdf.py --setup-rashi
```

This downloads `heb_rashi.traineddata` from the [Pninim project](https://gitlab.com/pninim.org/tessdata_heb_rashi). After setup, the OCR extractor will automatically use it.

## Output

For each input PDF, the tool produces:
- `<filename>_hebrew.txt` — extracted Hebrew text
- `<filename>_english.txt` — English translation

Use `--output-dir` to write all output to a specific directory.

## License

MIT
