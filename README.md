# Khojki OCR

I built a local pipeline from PDF/page image to detected lines, left-to-right recognition and editable UTF-8 Khojki drafts. The delivered model scores **23.30% character error rate (CER) on five annotated printed regions extracted from raw pages** and **20.39% on their manual crops**. Neither has an exact line. This is a review tool, not reliable automatic transcription.

## Run Locally

The modified model and its hash-bound input contract are included. Keep [models/khojki-printed.safetensors](models/khojki-printed.safetensors) and [models/khojki-printed.input.json](models/khojki-printed.input.json) together. A fresh Conda-forge environment was checked on Intel macOS; Conda or Micromamba is required for this recipe.

```sh
git clone https://github.com/aruhant/khojki-ocr.git
cd khojki-ocr
conda env create -f environment.yml
conda activate khojki-ocr
python -m pip install -e '.[ocr,test]'
python -m pip install --force-reinstall --no-deps --only-binary=:all: Pillow==13.0.0
python -m pip check
python -m khojki_ocr.server --model models/khojki-printed.safetensors --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Upload a PNG/JPEG or PDF, inspect the source beside streamed drafts, edit and download UTF-8. Edits persist in this browser and remain separate from raw predictions; neither is certified. Fonts and icons are served locally, without a CDN.

Cancellation takes effect between processing stages/lines, not during an active inference call. Uploads and reports stay in ignored `runs/web-jobs`; scans are not sent to cloud services. The default upload limit is 20 MB/20 pages.

Pillow must report RAQM support to regenerate shaped synthetic examples: `python -c 'from PIL import features; assert features.check("raqm")'`. The explicit wheel install above supplied it in the checked environment.

## Command Line

```sh
python -m khojki_ocr.pipeline --input scan.pdf --output runs/page \
  --model models/khojki-printed.safetensors --dpi 180 \
  --disable-references --export-drafts
```

The report contains raw predictions, coordinates, model/contract hashes and failed line IDs. `--export-drafts` writes nonempty, valid raw UTF-8 to `runs/page/drafts`, **not approved transcripts**. Empty rows can be recognition failures; always consult the report. Blank pages produce no text artifact. Errors are isolated so later lines continue; a completed run with errors returns exit code 1. Add `--segment-only` for detection alone. Exact-raster reference replay is a separate, explicitly enabled workflow, not OCR.

## Measured Results

The final checkpoint was selected by **1/46 development errors**, before its test evaluations. Training used 23 reviewed crops/302 code points from pages 17/20/21/26; development used six partial spans from page 22. Test pages 18/19/23/24 were excluded. The final stage completed 88 CPU updates in 1136.6 seconds; step 5 won and ties retained it. Training error was 4/302, not a test score. See [results/final-model-summary.json](results/final-model-summary.json).

| Manual Crop | Errors / Reference Code Points | CER |
| --- | --- | --- |
| `p018-row05` | 2 / 16 | 12.50% |
| `p018-row06` | 3 / 21 | 14.29% |
| `p018-row08` | 2 / 21 | 9.52% |
| `p018-row14` | 6 / 22 | 27.27% |
| `p019-row01` | 8 / 23 | 34.78% |

Manual crops total **21/103 errors**: 12 substitutions, six insertions, three deletions; **0/5 exact**. [results/final-crop-test.json](results/final-crop-test.json) retains references and predictions; [results/final-transcripts/README.md](results/final-transcripts/README.md) links their raw UTF-8 files. All six O signs (U+11232) were output as E+AA (U+11230 U+1122C), sequences that NFC does not unify. The alignment counts six O-to-AA substitutions alongside inserted/substituted E signs; visual recognition versus semantic Unicode mapping was not isolated.

Raw pages 18/19 took **154.77 seconds**, with 87 detected regions and six oversized-line failures. All five annotated regions matched geometrically, scoring **24/103 errors (23.30%)**, 0/5 exact, in [results/final-page-test.json](results/final-page-test.json). This is end-to-end **annotated-region CER, not full-page CER**; unlabelled regions are not automatically false positives. The full diagnostic [results/final-page-report.json](results/final-page-report.json) accompanies unverified [results/final-pages/page-0018.txt](results/final-pages/page-0018.txt) and [results/final-pages/page-0019.txt](results/final-pages/page-0019.txt).

A separate five-word, 20-code-point test on pages 23/24 scored **4/20 errors (20%), 1/5 exact**: [results/final-fresh-words.json](results/final-fresh-words.json). It favours short, legible words and is too small for a robust claim. The historical five lines were reused for diagnostics during earlier iterations; references were prediction-free, AI-assisted and chart-adjudicated, not expert-certified. One fresh reviewer incidentally saw aggregate metrics, but not prediction text. Neither set is a strong blind benchmark.

CER uses NFC-normalized code points, counting punctuation and whitespace and retaining incorrect output. Raw CLI/API exports enforce target-only validation; browser-edited drafts are unrestricted, unverified UTF-8. There is no trained script classifier or dictionary correction. Raw input remains default: final normalization tied 1/46 development errors; an earlier same-model comparison worsened 1/41 to 2/41. There is **no handwriting training or CER result**.

## Reproduce

Private exercise scans and crops are not distributed. Restore the original exercise PDF as `data/private/sample.pdf`. Historical test previews used macOS PDFKit thumbnails, not PDFium; [scripts/render_reference_previews.swift](scripts/render_reference_previews.swift) reproduced their exact PNG hashes and RGB pixels:
```sh
swift scripts/render_reference_previews.swift data/private/sample.pdf data/private 17 18 19 20
```
The later previews use PDFium at a direct scale, without rounding/resizing the resulting image. Both recipes refuse overwrites:
```sh
python - <<'PY'
from pathlib import Path
import pypdfium2 as pdfium
with pdfium.PdfDocument("data/private/sample.pdf") as pdf:
  for number in (21, 22, 23, 24, 26):
    page = pdf[number - 1]
    bitmap = page.render(scale=2000 / page.get_width())
    path = Path("data/private") / f"preview-{number:03d}.png"
    with path.open("xb") as stream:
      bitmap.to_pil().convert("RGB").save(stream, format="PNG")
    bitmap.close()
    page.close()
PY
```
The manifests [data/fresh-printed-references.json](data/fresh-printed-references.json), [data/training-final.json](data/training-final.json) and [data/development-final.json](data/development-final.json) record the crop bounds; adaptation checks their source hashes. These commands repeat evaluation, not the complete training chain; intermediate training checkpoints are not bundled:
```sh
python -m khojki_ocr.evaluation --references data/references.json \
  --model models/khojki-printed.safetensors --output runs/check.json
python -m khojki_ocr.evaluation --references data/fresh-printed-references.json \
  --model models/khojki-printed.safetensors --output runs/fresh-check.json
python -m khojki_ocr.pipeline --input data/private/sample.pdf --pages 18,19 \
  --dpi 180 --model models/khojki-printed.safetensors --disable-references \
  --export-drafts --output runs/page-check
# Run this even if the preceding command reports isolated line errors.
python -m khojki_ocr.evaluation --references data/references.json \
  --page-report runs/page-check/report.json --output runs/page-check/cer.json
python -m pytest -q
```

See [RETRO.md](RETRO.md) for failed experiments, Unicode choices, label budget and CPU scaling. Older reports are retained as historical experiments; they do not describe the bundled model.

The 154.77-second wall time was measured with external `/usr/bin/time -p` around the two-page CLI invocation, including model loading, inference and report output.

The modified recognizer derives from the Apache-2.0 Kraken PP-OCRv6 small base model. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) retains its attribution/license and the offline assets' notices, including Noto Sans Khojki's SIL Open Font License. Those licenses do not grant rights to private scans.