from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.viz.mask import make_mask_frames, make_overlay_frames
from ms_model.viz.video import frames_to_video

if __name__ == '__main__':
    print("Début du mask_test")

    npz_path = "../../EVOwithMS/datasets/EV-IMO/eval/tabletop/npz/seq_00.npz"
    ea = load_events(npz_path)
    fm = load_evimo_mask(npz_path)
    print(f"{len(ea)} events, {len(fm)} masques GT")

    mask_frames = make_mask_frames(fm)
    frames_to_video(mask_frames, "seq_00_mask.mp4", fps=20)
    print("Vidéo masque seul écrite : seq_00_mask.mp4")

    overlay_frames = make_overlay_frames(ea, fm)
    frames_to_video(overlay_frames, "seq_00_overlay.mp4", fps=20)
    print("Vidéo events + masque écrite : seq_00_overlay.mp4")

    print("Fin du mask_test")
