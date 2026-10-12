# ssl_probing

## Download LibriTTS-R with ModelScope

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the `dev.clean` split:

```bash
python download_data.py
```

The script is equivalent to:

```python
from modelscope.msdatasets import MsDataset

dataset = MsDataset.load(
	"modelscope/libritts_r",
	subset_name="default",
	split="dev.clean",
)
```

To download another split, for example `train.clean.100`:

```bash
python download_data.py --split train.clean.100
```

## Prepare LibriTTS for MFA

Install Montreal Forced Aligner separately, then prepare the downloaded
archive. The script reads LibriTTS normalized transcripts and writes one
`.lab` file next to each WAV file:

```bash
python prepare_mfa.py \
	--archive path/to/train-clean-100.tar.gz \
	--corpus-dir data/libritts_mfa
```

Download the MFA models and align the prepared corpus:

```bash
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
mfa validate data/libritts_mfa english_us_mfa english_mfa
mfa align data/libritts_mfa english_us_mfa english_mfa data/libritts_aligned
```

The aligned TextGrid files in `data/libritts_aligned` provide the phoneme
time boundaries needed to map SpeechTokenizer frames to phoneme labels.

## Probe SpeechTokenizer layers

Run the probe on any LibriTTS WAV file:

```bash
python probe.py \
	--input data/libritts_mfa/LibriTTS/train-clean-100/123/456/123-456-0001.wav \
	--output-dir probe_outputs
```

The script loops over every RVQ layer returned by `model.encode()`. For each
layer it saves cumulative token codes as `.npy` and a reconstruction as `.wav`.

## Test TensorBoard

Generate fake layer-wise probing metrics:

```bash
python write_fake_tensorboard.py
```

Start TensorBoard:

```bash
tensorboard --logdir runs
```