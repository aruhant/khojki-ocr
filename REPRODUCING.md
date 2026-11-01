# Reproduce the Evaluation

These steps are optional and are not needed to run the app. Complete [README.md](README.md)'s setup first and activate the `khojki-ocr` environment.

## Check the Included Sample

This generated sample needs no private PDF:

```sh
python -m khojki_ocr.pipeline --input web/assets/printed-sample.png \
  --model models/khojki-printed.safetensors --disable-references \
  --export-drafts --output runs/sample-check
```

Expected text and measured cold-run results are in [results/cold-run-validation.json](results/cold-run-validation.json).

## Prepare the Reference Images

The private exercise scans and crops are not included in the repository. To repeat the recorded benchmarks, place the original 47-page exercise PDF at `data/private/sample.pdf`. An arbitrary PDF will not match the reference manifests.

Historical reference previews for pages 17/18/19/20 were rendered with macOS PDFKit. On macOS with Swift installed, run:

```sh
swift scripts/render_reference_previews.swift data/private/sample.pdf data/private 17 18 19 20
```

Later previews use PDFium. Run this command to generate the evaluation and research previews:

```sh
python - <<'PY'
from pathlib import Path
import pypdfium2 as pdfium
with pdfium.PdfDocument("data/private/sample.pdf") as pdf:
  for number in (21, 22, 23, 24, 26, 28, 29, 31, 32, 34, 44):
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

Once processing finishes, score the report:

```sh
python -m khojki_ocr.evaluation --references data/references.json \
  --page-report runs/page-check/report.json --output runs/page-check/cer.json
```

Compare the outputs with [results/final-crop-test.json](results/final-crop-test.json), [results/final-fresh-words.json](results/final-fresh-words.json) and [results/cold-page-test.json](results/cold-page-test.json). The current page report, including raw recognized text, is [results/cold-page-report.json](results/cold-page-report.json). Older page reports and text files remain unchanged as historical records.

## Run the Tests

Install the test dependencies before running the suite. The Pillow wheel installed during setup supplies RAQM shaping support; the check below verifies it for synthetic rendering tests.

```sh
python -m pip install -e '.[test]'
python -c 'from PIL import features; assert features.check("raqm"), "RAQM support is required for shaping tests"'
python -m pytest -q
```

## Fine-Tuning Experiments

Training requires reviewed source crops and a trusted Kraken `.ckpt` initialization file, neither of which is bundled. Complete the preview preparation above and supply your checkpoint path before running:

```sh
python -m khojki_ocr.adapt --training data/training-research-002.json \
  --development data/development-research-002.json \
  --checkpoint path/to/trusted.ckpt --output runs/new-fine-tune \
  --lrate 0.0002 --max-steps 100 --seconds 1200
```

Add `--augment` for a separate image-augmentation experiment. Checkpoints are selected by development CER; held-out pages and unresolved labels are rejected. Use a new output directory for each run. [results/research-experiments-001.json](results/research-experiments-001.json) records the initial experiments; neither candidate was promoted because real-page performance did not improve.

The frozen research manifests contain 24 training crops/308 code points and eight development spans/54 code points. Their inherited `scope` fields describe the earlier base corpus; the `lines` arrays and experiment record contain the expanded counts. Preserve the manifests used by each run rather than rewriting its inputs afterward.

Use `--segmentation-mode components` to require component-based rows, or `--segmentation-mode baselines` to test the learned detector. The default `auto` keeps the current component-first behavior. These modes enable handwriting experiments; they do not supply a handwriting-trained recognizer.

Prepare unlabelled source crops without loading a recognizer:

```sh
python -m khojki_ocr.pipeline --input data/private/sample.pdf \
  --pages 20,21,22,28,29,31,32,34,44 --segment-only \
  --segmentation-mode components --line-crops runs/new-annotation-crops \
  --output runs/new-annotation-pack
```

The recorded run produced 303 candidate crops. Review the complete pages for omitted lines, crop boundaries and reading order, then transcribe and classify each region before admitting it to a split. Page-disjoint splits within this PDF do not establish generalization across books or writers.

## Measurement Notes

The final checkpoint was selected on development data before its test evaluations: 1/46 errors on six partial spans from page 22. Training used 23 reviewed crops/302 code points from pages 17/20/21/26; test pages 18/19/23/24 were excluded. The final stage ran 88 CPU updates in 1136.6 seconds and selected step 5. Training error was 4/302. These details are recorded in [results/final-model-summary.json](results/final-model-summary.json); intermediate checkpoints for the complete training chain are not bundled.

CER counts NFC-normalized code points, including punctuation and whitespace, without discarding incorrect predictions. The manual-crop result is 21/103 errors: 12 substitutions, six insertions and three deletions. All six reference O signs (U+11232) were predicted as E+AA (U+11230 U+1122C); NFC does not unify those sequences. Visual recognition versus Unicode mapping was not isolated.

The current raw-page result is 21/103 errors across the five geometrically matched annotated regions, with 40 total detections and zero recognition failures. It is not full-page CER. The two-page cold CLI took 106.35 seconds on the Intel macOS CPU, timed around the fresh process including model loading and report output. The recognizer weights are unchanged; component-based line extraction handles printed borders, and oversized rows are split at blank columns and rejoined without extra line breaks.

The five historical lines were reused for diagnostics in earlier iterations. References were transcribed without prediction text, using AI assistance and independent chart review, not expert certification. The fresh sample favours short, legible words; one reviewer incidentally saw aggregate metrics, but not predictions. Neither sample is a strong blind benchmark.

Raw input is the default: final normalization tied the 1/46 development error count; an earlier same-model comparison worsened from 1/41 to 2/41. No dictionary correction, trained script classifier or handwriting benchmark is included. Older reports describe earlier experiments and pipeline versions.