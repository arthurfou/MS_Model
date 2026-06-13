from pathlib import Path

import cv2
import numpy as np


def frames_to_video(frames: list[np.ndarray], out_path: str | Path, fps: int = 20, fourcc: str = "mp4v") -> Path:
    """Écrit les frames dans une vidéo (cv2.VideoWriter) et retourne le chemin.

    Le conteneur est déduit de l'extension de out_path (ex: .mp4) et doit être
    compatible avec le codec choisi via fourcc (ex: "mp4v" = MPEG-4 Part 2,
    supporté nativement par le build OpenCV/ffmpeg de l'env devo — "avc1"/H.264
    n'a pas d'encodeur disponible ici).
    """
    out_path = Path(out_path)
    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    return out_path