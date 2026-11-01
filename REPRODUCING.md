# Reproduce the Evaluation

These steps are optional and are not needed to run the app. Complete [README.md](README.md)'s setup first and activate the `khojki-ocr` environment.

## Prepare the Reference Images

The private exercise scans and crops are not included in the repository. To repeat the recorded benchmarks, place the original 47-page exercise PDF at `data/private/sample.pdf`. An arbitrary PDF will not match the reference manifests.

Historical reference previews for pages 17/18/19/20 were rendered with macOS PDFKit. On macOS with Swift installed, run:

```sh
swift scripts/render_reference_previews.swift data/private/sample.pdf data/private 17 18 19 20
```

Later previews use PDFium. Run this command to generate pages 21/22/23/24/26:

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

Use these exact renderers and scales: crop bounds and source hashes depend on them. Do not round or resize PDFium's output; some pages are 2001 pixels wide. Both commands refuse to overwrite existing previews.

## Evaluate the Model

Evaluate the five manual crops and the separate short-word sample:

```sh
python -m khojki_ocr.evaluation --references data/references.json \
  --model models/khojki-printed.safetensors --output runs/check.json
python -m khojki_ocr.evaluation --references data/fresh-printed-references.json \
  --model models/khojki-printed.safetensors --output runs/fresh-check.json
```

For the raw-page evaluation, first process pages 18/19:

```sh
python -m khojki_ocr.pipeline --input data/private/sample.pdf --pages 18,19 \
  --dpi 180 --model models/khojki-printed.safetensors --disable-references \
  --export-drafts --output runs/page-check
```

This benchmark has six oversized-line failures outside the annotated regions, so the pipeline exits with code 1 but still writes its report. Once it finishes, score that report:

```sh
python -m khojki_ocr.evaluation --references data/references.json \
  --page-report runs/page-check/report.json --output runs/page-check/cer.json
```

Compare the outputs with [results/final-crop-test.json](results/final-crop-test.json), [results/final-fresh-words.json](results/final-fresh-words.json) and [results/final-page-test.json](results/final-page-test.json). The recorded page report is [results/final-page-report.json](results/final-page-report.json). Raw text is in [results/final-transcripts/README.md](results/final-transcripts/README.md) and [results/final-pages/page-0018.txt](results/final-pages/page-0018.txt) / [results/final-pages/page-0019.txt](results/final-pages/page-0019.txt).

## Run the Tests

Install the test dependencies before running the suite. The Pillow wheel installed during setup supplies RAQM shaping support; the check below verifies it for synthetic rendering tests.

```sh
python -m pip install -e '.[test]'
python -c 'from PIL import features; assert features.check("raqm"), "RAQM support is required for shaping tests"'
python -m pytest -q
```

## Measurement Notes

The final checkpoint was selected on development data before its test evaluations: 1/46 errors on six partial spans from page 22. Training used 23 reviewed crops/302 code points from pages 17/20/21/26; test pages 18/19/23/24 were excluded. The final stage ran 88 CPU updates in 1136.6 seconds and selected step 5. Training error was 4/302. These details are recorded in [results/final-model-summary.json](results/final-model-summary.json); intermediate checkpoints for the complete training chain are not bundled.

CER counts NFC-normalized code points, including punctuation and whitespace, without discarding incorrect predictions. The manual-crop result is 21/103 errors: 12 substitutions, six insertions and three deletions. All six reference O signs (U+11232) were predicted as E+AA (U+11230 U+1122C); NFC does not unify those sequences. Visual recognition versus Unicode mapping was not isolated.

The raw-page result is 24/103 errors across the five geometrically matched annotated regions, with 87 total detections. It is not full-page CER. The two-page CLI took 154.77 seconds on the Intel macOS CPU, measured by external `/usr/bin/time -p`, including model loading and report output.

The five historical lines were reused for diagnostics in earlier iterations. References were transcribed without prediction text, using AI assistance and independent chart review, not expert certification. The fresh sample favours short, legible words; one reviewer incidentally saw aggregate metrics, but not predictions. Neither sample is a strong blind benchmark.

Raw input is the default: final normalization tied the 1/46 development error count; an earlier same-model comparison worsened from 1/41 to 2/41. No dictionary correction, trained script classifier or handwriting benchmark is included. Older reports are retained as historical experiments, not measurements of the bundled model.