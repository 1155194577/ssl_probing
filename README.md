# ssl_probing

## Download LibriTTS-R with ModelScope

Install ModelScope:

```bash
python -m pip install modelscope
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