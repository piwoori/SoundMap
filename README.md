# YAMNet Quick Test

## 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run
```bash
python scripts/yamnet_quickcheck.py path/to/your_audio.wav --topk 5
```

- Accepts WAV/MP3/M4A, auto‑resamples to 16kHz mono.
- Prints Top‑K labels with probabilities.
- 테스트 코드