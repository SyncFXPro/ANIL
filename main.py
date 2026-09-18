import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity works the same in 3-D or (44100,)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return float(np.dot(a, b) / denom)


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
    f1 = np.array([1.0, 2.0, 3.0])
    f2 = np.array([1.2, 1.8, 2.9])

    similarity = cosine_similarity(f1, f2)

    print("F1:", f1)
    print("F2:", f2)
    print("Cosine similarity:", similarity)
    print("Close to 1: almost the same direction.")
    print("Around 0: very different directions.")
    print("Near -1: opposite directions.")

    plot_vectors_3d(f1, f2)


if __name__ == "__main__":
    main()
