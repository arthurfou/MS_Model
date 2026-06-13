from ms_model.io.loaders import load_events
from ms_model.viz.render import make_frames
from ms_model.viz.video import frames_to_video
from pathlib import Path

if __name__ == '__main__':
    print("Début du video_test")

    ea = load_events("../../EVOwithMS/datasets/EV-IMO/eval/tabletop/npz/seq_00.npz")
    print(f"{len(ea)} events, résolution {ea.H}x{ea.W}, {len(ea.frame_ts)} frames GT")

    frames = make_frames(ea)
    print(f"{len(frames)} frames rendues, shape={frames[0].shape}")

    out_path = frames_to_video(frames, "seq_00.mp4", fps=20)
    print(f"Vidéo écrite : {out_path.absolute()}")

    print("Fin du video_test")
