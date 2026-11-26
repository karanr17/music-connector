import os
import threading
from dataclasses import dataclass

import numpy as np
import torch

from .audio import windows
from .config import MODEL_ID, SAMPLE_RATE

# The text tokenizer's thread pool trips a warning on every later ffmpeg fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SongEmbedding:
    song_vec: np.ndarray
    window_vecs: np.ndarray
    window_starts: list[float]
    spread: float
    duration_s: float


def pool(window_vecs: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean-pool unit window vectors into a unit song vector, plus how far windows stray from it."""
    mean = window_vecs.mean(axis=0)
    song_vec = mean / np.linalg.norm(mean)
    spread = float(np.mean(1.0 - window_vecs @ song_vec))
    return song_vec.astype(np.float32), spread


class Embedder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, batch_size: int = 1):
        from muq import MuQMuLan

        self.model_id = model_id
        self.device = device or best_device()
        self.batch_size = batch_size
        self.model = MuQMuLan.from_pretrained(model_id).to(self.device).eval()
        # MuQ-MuLan on MPS isn't safe to call from several threads at once, and `mc serve` runs
        # live capture and /analyze requests in worker threads.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def embed_windows(self, wins: np.ndarray) -> np.ndarray:
        """Embed [n, samples] windows → [n, 512] unit vectors.

        Calls the MuLan audio tower directly: the public `MuQMuLan.__call__` averages all clips
        of a song into one vector, and we need the per-window vectors too.
        """
        out = []
        with self._lock:
            for i in range(0, len(wins), self.batch_size):
                batch = torch.from_numpy(wins[i : i + self.batch_size]).to(self.device)
                out.append(self.model.mulan_module.get_audio_latents(batch).float().cpu())
        return torch.cat(out).numpy()

    def embed_wav(self, wav: np.ndarray) -> SongEmbedding:
        wins, starts = windows(wav)
        vecs = self.embed_windows(wins)
        song_vec, spread = pool(vecs)
        return SongEmbedding(song_vec, vecs, starts, spread, len(wav) / SAMPLE_RATE)

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self.model(texts=texts).float().cpu().numpy()

    def sync(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
