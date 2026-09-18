import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity works the same in 3-D or (44100,)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return float(np.dot(a, b) / denom)

def record_audio(duration: float = 5.0) -> np.ndarray:
    print(f"Recording audio for {duration} seconds...")

    F = sd.rec(
        int(duration * 44100),
        samplerate=44100,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")
    return F.flatten()


def plot_vectors_3d(f1: np.ndarray, f2: np.ndarray) -> None:
    """Show only the first 3 dimensions. A real sound vector cannot be plotted in full."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    origin = np.zeros(3)
    ax.quiver(*origin, *f1, color="C0", label="F1")
    ax.quiver(*origin, *f2, color="C1", label="F2")

    max_val = float(np.max(np.abs(np.stack([f1, f2])))) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(0, max_val)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend()
    plt.show()


def main() -> None:
    # Two sound-like vectors (later these can be waveform samples)
    f1 = record_audio()
    f2 = np.convolve(f1, [0.25, 0.5, 0.25], mode="same")
    similarity = cosine_similarity(f1, f2)

    print("F1:", f1)
    print("F2:", f2)
    print("Cosine similarity:", similarity)


if __name__ == "__main__":
    main()
