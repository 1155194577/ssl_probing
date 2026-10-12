"""Probe every residual-vector-quantizer layer of SpeechTokenizer."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from modelscope import snapshot_download
from scipy.io.wavfile import read, write
from scipy.signal import resample_poly
from speechtokenizer import SpeechTokenizer


MODEL_NAME = "openmoss/SpeechTokenizer"


def load_audio(audio_path: Path, sample_rate: int) -> torch.Tensor:
    original_sample_rate, audio = read(str(audio_path))
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    if original_sample_rate != sample_rate:
        divisor = math.gcd(original_sample_rate, sample_rate)
        audio = resample_poly(
            audio,
            sample_rate // divisor,
            original_sample_rate // divisor,
            axis=1,
        )
    waveform = torch.from_numpy(np.ascontiguousarray(audio))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.unsqueeze(0)


def save_waveform(waveform: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    audio = waveform.squeeze().detach().cpu().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    write(output_path, sample_rate, (audio * 32767).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode audio and probe every SpeechTokenizer RVQ layer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input LibriTTS WAV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe_outputs"),
        help="Directory for per-layer outputs (default: probe_outputs).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Use an existing SpeechTokenizer model directory.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input audio does not exist: {args.input}")

    model_dir = args.model_dir or Path(snapshot_download(MODEL_NAME))
    config_path = model_dir / "speechtokenizer_hubert_avg" / "config.json"
    checkpoint_path = model_dir / "speechtokenizer_hubert_avg" / "SpeechTokenizer.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeechTokenizer.load_from_checkpoint(
        str(config_path),
        str(checkpoint_path),
    ).to(device)
    model.eval()

    waveform = load_audio(args.input, model.sample_rate).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        codes = model.encode(waveform)
        layer_count = codes.shape[0]
        for layer_index in range(layer_count):
            layer_number = layer_index + 1
            layer_codes = codes[:layer_number]
            np.save(
                args.output_dir / f"layer_{layer_number:02d}_codes.npy",
                layer_codes.cpu().numpy(),
            )
            reconstructed = model.decode(layer_codes)
            save_waveform(
                reconstructed,
                model.sample_rate,
                args.output_dir / f"layer_{layer_number:02d}_reconstruction.wav",
            )
            print(
                f"Layer {layer_number}/{layer_count}: "
                f"saved cumulative codes and reconstruction"
            )


if __name__ == "__main__":
    main()