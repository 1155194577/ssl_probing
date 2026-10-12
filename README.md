# ssl_probing

## Download LibriTTS `train-clean-100`

Download the archive directly from OpenSLR. No TensorFlow installation is
required:

```bash
python download_libritts.py
```

The default output path is `data/train-clean-100.tar.gz`. To choose another
path:

```bash
python download_libritts.py --output /path/to/train-clean-100.tar.gz
```