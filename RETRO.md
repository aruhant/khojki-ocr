# Technical Retro

PDFium renders pages, Kraken BLLA detects lines, and PP-OCRv6 recognizes them after fine-tuning on 23 printed crops. Five test lines (103 code points) score 23.30% CER end to end and 20.39% on manual crops.

## Error Breakdown by Stage

- **Line detection:** all five annotated lines were found without splits. Raw pages gave 24 text errors versus 21 on manual crops; the net increase of three also includes differences in crop geometry and raster resolution.
- **Recognition and Unicode mapping:** manual crops had 12 substitutions, six insertions and three deletions. All six O signs (U+11232) were output as E+AA (U+11230 U+1122C); NFC does not unify them. The model emits Unicode directly, so these stages were not measured separately.
- **Common errors:** three inserted TTHA letters, independent vowel I misread both times and A read as AA.

## Experiments That Did Not Work

- **Synthetic-only training:** 180 font-rendered lines reached 43.7% accuracy on synthetic validation but 98.06% CER on the real test lines.
- **Default variable-width inference:** training pads lines to a fixed width; padding at inference the same way cut CER from 98.06% to 89.32% with the same weights.
- **Mixing synthetic and real lines:** adding 323 synthetic lines to 17 real lines kept development errors at 2/41.
- **Background flattening:** development errors rose from 1/41 to 2/41 and tied at 1/46 on the final model, so images stay unmodified.

## Encoding Decisions

- Output is NFC Unicode in U+11200–U+11241 (excluding unassigned U+11212), plus space and newline. Each prediction is checked against this set; invalid characters are reported and counted in CER, and the CLI writes only drafts that pass.
- Dependent vowel signs U+1122C–U+11233 (AA, I, II, U, E, AI, O, AU) follow consonants. Independent vowels use U+11200–U+11207; anusvara uses U+11234 and the word separator U+1123A, not an ASCII colon or space.
- Khojki has no Unicode digits, and none appear in the labelled lines, so digits are excluded; borrowed Gujarati digits (U+0AE6–U+0AEF) would need to be added explicitly.

## Bottleneck

More labelled real lines. Fine-tuning the synthetic model on 17 real lines cut CER from 89.32% to 28.16%. Common errors involve O (two training examples) or independent I (none). I estimate about 500 printed lines from several books, with each vowel sign appearing a few dozen times, then about 200 lines per handwriting style.

## Scale

At 180 DPI, two pages took 154.77 seconds on a 4-core Intel Core i7-10510U, including model loading: about 77 seconds per page, or one hour for 45 pages, with no paid compute. Handwriting needs real handwritten labels, detection checks on uneven baselines and writer-disjoint evaluation. Labelling becomes the main cost.

## AI Tools

An AI draft read vowel sign AA plus anusvara (a small open circle) as vowel sign II (U+1122E). Enlarging the crop and comparing both shapes with the script chart exposed the error; the label is now U+1122C U+11234.
