"""Eight-example memorization check; scores are training fit, never generalization."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

from khojki_ocr.core import POLICY_VERSION, validate_text
from khojki_ocr.evaluation import score_line
from khojki_ocr.training import file_sha256, load_font, render_line, write_input_contract, write_json

CASES = (
    "\U00011208\U0001122c\U00011219\U0001122d",
    "\U0001121e\U0001122e\U00011226\U00011230",
    "\U00011224\U0001122f\U0001121f\U0001122c",
    "\U00011229\U00011232\U0001122a\U00011230\U00011234",
    "\U00011200\U0001121f\U0001121e\U0001122d",
    "\U00011210\U0001122c\U00011219\U0001122c\U00011234",
    "\U00011226\U00011230\U0001120e\U0001122e\U0001123a",
    "\U0001120a\U0001122f\U00011219\U00011208\U0001122c\U0001123a",
)


def prepare_examples(output: Path, font_path: Path) -> dict:
    from PIL import Image, ImageDraw

    if output.exists():
        raise FileExistsError(output)
    font, supported, renderer = load_font(font_path, 48)
    records = []
    for index, text in enumerate(CASES):
        validate_text(text)
        if not {ord(character) for character in text} <= supported:
            raise ValueError("A diagnostic character is missing from the font")
        image = render_line(text, font)
        if math.ceil(image.width * 128 / image.height) + 32 > 512:
            raise ValueError("Diagnostic line exceeds the verified static width")
        records.append({"id": f"case-{index + 1}", "text": text, "size": list(image.size), "image": image})
    output.mkdir(parents=True)
    sheet = Image.new("RGB", (900, len(records) * 130), "white")
    drawing = ImageDraw.Draw(sheet)
    paths = []
    for index, record in enumerate(records):
        image = record.pop("image")
        path = output / f"{record['id']}.png"
        image.save(path)
        path.with_suffix(".gt.txt").write_text(record["text"], encoding="utf-8")
        record["image"] = str(path.resolve())
        record["sha256"] = file_sha256(path)
        paths.append(str(path.resolve()))
        sheet.paste(image, (20, index * 130 + 22))
        drawing.text((380, index * 130 + 24), record["id"], fill="black")
        drawing.text((380, index * 130 + 45), " ".join(f"{ord(character):05X}" for character in record["text"]), fill="black")
        image.close()
    sheet.save(output / "review.png")
    sheet.close()
    (output / "same-eight.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    manifest = {"policy_version": POLICY_VERSION, "purpose": "memorization_only",
                "synthetic": True, "real_training_lines": 0, "renderer": renderer,
                "font_sha256": file_sha256(font_path), "max_width": 512, "lines": records}
    write_json(output / "examples.json", manifest)
    return manifest


def run_check(examples: Path, checkpoint: Path, output: Path, seconds: int = 900) -> dict:
    if not 1 <= seconds <= 900:
        raise ValueError("The learning check is capped at 900 seconds")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((examples / "examples.json").read_text())
    if manifest["purpose"] != "memorization_only" or len(manifest["lines"]) != 8:
        raise ValueError("Exactly eight diagnostic examples are required")
    for record in manifest["lines"]:
        image_path = Path(record["image"])
        if file_sha256(image_path) != record["sha256"] or image_path.with_suffix(".gt.txt").read_text() != record["text"]:
            raise ValueError("Reviewed image/label pairs changed")
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    import torch
    from lightning.pytorch import Callback, seed_everything
    from kraken.configs import RecognitionInferenceConfig
    from kraken.models.convert import convert_models
    from kraken.train import KrakenTrainer
    from kraken.train.ppocr import PPOCRv6RecognitionModel

    seed_everything(42, workers=True)
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    started = time.monotonic()
    history = []
    expected = [record["text"] for record in manifest["lines"]]
    state = {"stop_reason": None, "gradient_norm": None, "error": None}
    decoder = RecognitionInferenceConfig().decoder

    class FitObserver(Callback):
        def measure(self, trainer, module):
            was_training = module.net.training
            module.net.eval()
            records = []
            input_shapes = []
            with torch.inference_mode():
                for batch in trainer.datamodule.val_dataloader():
                    if not torch.isfinite(batch["image"]).all():
                        raise ValueError("Non-finite model input")
                    input_shapes.append(list(batch["image"].shape))
                    logits, lengths = module.net(batch["image"], batch["seq_lens"])
                    decoded = decoder(logits.softmax(1).squeeze(2), lengths)
                    offset = 0
                    for positions, target_length in zip(decoded, batch["target_lens"].tolist()):
                        encoded = [(int(label), 0, 0, 0.0) for label in batch["target"][offset:offset + target_length]]
                        reference = "".join(item[0] for item in module._val_codec.decode(encoded))
                        prediction = "".join(item[0] for item in module.codec.decode(positions))
                        records.append(score_line(reference, prediction))
                        offset += target_length
            module.net.train(was_training)
            if sorted(record["reference_text"] for record in records) != sorted(expected):
                raise ValueError("Labels failed their actual dataloader/codec round trip")
            records.sort(key=lambda record: expected.index(record["reference_text"]))
            total = sum(record["reference_characters"] for record in records)
            errors = sum(record["errors"] for record in records)
            loss = trainer.callback_metrics.get("train_loss_step")
            item = {"step": trainer.global_step, "elapsed_seconds": round(time.monotonic() - started, 3),
                    "cer": errors / total, "errors": errors, "reference_characters": total,
                    "loss": float(loss) if loss is not None else None,
                    "gradient_norm": state["gradient_norm"], "input_shapes": input_shapes,
                    "predictions": records}
            history.append(item)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=True) + "\n")
            print(json.dumps({key: value for key, value in item.items() if key != "predictions"}), flush=True)
            if item["cer"] <= 0.01:
                state["stop_reason"] = "target_training_cer"
                trainer.should_stop = True

        def on_train_start(self, trainer, module):
            self.measure(trainer, module)

        def on_after_backward(self, trainer, module):
            squares = sum(float(parameter.grad.detach().float().square().sum())
                          for parameter in module.parameters() if parameter.grad is not None)
            norm = math.sqrt(squares)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Non-finite or zero training gradients")
            state["gradient_norm"] = norm

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.global_step == 1 or trainer.global_step % 5 == 0:
                self.measure(trainer, module)
            if time.monotonic() - started >= seconds:
                state["stop_reason"] = "time_limit"
                trainer.should_stop = True

    image_paths = [record["image"] for record in manifest["lines"]]
    options = dict(training_data=image_paths,
                   evaluation_data=image_paths, partition=1,
                   format_type="path", normalization="NFC", normalize_whitespace=False,
                   bidi_reordering=False, augment=False, num_workers=0, max_width=512,
                   variant="small", resize="fail", batch_size=8, optimizer="AdamW", lrate=0.0005,
                   schedule="constant", warmup=0, quit="fixed", epochs=500, min_epochs=0,
                   checkpoint_path=str(output.resolve()))
    config = PPOCRv6RecognitionModel._config_class(**options)
    data_config = PPOCRv6RecognitionModel._data_config_class(**options)
    data = PPOCRv6RecognitionModel._data_module_class(data_config)
    observer = FitObserver()
    trainer = KrakenTrainer(accelerator="cpu", devices=1, precision="32-true", min_epochs=0,
                            max_epochs=500, max_steps=500, enable_progress_bar=False,
                            enable_summary=False, enable_checkpointing=False, num_sanity_val_steps=0,
                            limit_val_batches=0, callbacks=[observer], gradient_clip_val=1.0,
                            deterministic="warn")
    model = PPOCRv6RecognitionModel.load_from_checkpoint(str(checkpoint), config=config, weights_only=False)
    write_json(output / "config.json", {"scope": "training_fit_only", "examples": manifest,
               "checkpoint_sha256": file_sha256(checkpoint), "options": options,
               "seconds": seconds, "target_cer": 0.01, "auxiliary_loss": "unchanged Kraken CTC + NRTR"})

    def interrupt(signum, frame):
        state["stop_reason"] = "time_limit" if signum == signal.SIGALRM else "interrupted"
        raise KeyboardInterrupt

    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGALRM, signal.SIGHUP, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(1, seconds - (time.monotonic() - started)))
    try:
        trainer.fit(model, data)
    except KeyboardInterrupt:
        state["stop_reason"] = state["stop_reason"] or "interrupted"
    except Exception as error:
        state["stop_reason"] = "error"
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if state["stop_reason"] != "error":
        observer.measure(trainer, model)
        trainer.save_checkpoint(output / "diagnostic.ckpt")
        convert_models([output / "diagnostic.ckpt"], output / "diagnostic.safetensors", weights_format="safetensors")
        write_input_contract(output / "diagnostic.safetensors", data_config.max_width, data_config.padding)
    summary = {**state, "scope": "training_fit_only_not_generalization",
               "steps": trainer.global_step, "elapsed_seconds": round(time.monotonic() - started, 3),
               "initial_cer": history[0]["cer"] if history else None,
               "final_cer": history[-1]["cer"] if history else None,
               "passed": bool(history and history[-1]["cer"] <= 0.01)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if args.font:
        prepare_examples(args.examples, args.font)
        return
    if args.checkpoint is None or args.output is None:
        parser.error("Training requires --checkpoint and --output")
    result = run_check(args.examples, args.checkpoint, args.output, args.seconds)
    if result["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()