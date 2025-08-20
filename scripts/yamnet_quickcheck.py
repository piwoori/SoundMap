# scripts/yamnet_quickcheck.py
# Usage:
#   python scripts/yamnet_quickcheck.py /path/to/audio.wav --topk 5
#
# Notes:
# - Accepts any audio file readable by librosa; will resample to 16kHz mono automatically.
# - Prints Top‑K YAMNet labels with probabilities based on clip-level average.
import argparse, numpy as np, tensorflow as tf, tensorflow_hub as hub, librosa, sys

def load_audio_16k(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    return y.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to audio file (wav/mp3/m4a...)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K labels to print")
    args = parser.parse_args()

    print("🔄 Loading YAMNet from TF Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode()
    classes = [l.strip() for l in tf.io.gfile.GFile(class_map_path).read().splitlines()]

    print(f"🎧 Loading audio: {args.audio_path}")
    wav = load_audio_16k(args.audio_path)

    scores, embeddings, spectrogram = model(wav)   # scores: [frames, 521]
    clip_scores = tf.reduce_mean(scores, axis=0).numpy()

    top_idx = clip_scores.argsort()[-args.topk:][::-1]
    print("\n✅ Top results:")
    for i in top_idx:
        print(f"{classes[i]}\t{clip_scores[i]:.3f}")

if __name__ == "__main__":
    main()