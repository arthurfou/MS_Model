from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask, remove_events_in_mask
from ms_model.viz.mask import make_overlay_frames
from ms_model.viz.render import make_frames
from ms_model.viz.video import frames_to_video

if __name__ == '__main__':
    print("Début du masking_test")

    npz_path = "../../EVOwithMS/datasets/EV-IMO/eval/tabletop/npz/seq_00.npz"
    ea = load_events(npz_path)
    fm = load_evimo_mask(npz_path)

    fm_thick = thicken_mask(fm, radius=3)

    ea_filtered = remove_events_in_mask(ea, fm_thick)
    print(f"events avant filtrage : {len(ea)}")
    print(f"events après filtrage : {len(ea_filtered)}")

    # vidéo des events restants (sans la zone masquée), pour vérifier visuellement
    frames = make_frames(ea_filtered, frame_ts=fm.ts)
    frames_to_video(frames, "seq_00_filtered.mp4", fps=20)
    print("Vidéo events filtrés écrite : seq_00_filtered.mp4")

    # comparaison overlay avec masque épaissi
    overlay_frames = make_overlay_frames(ea, fm_thick)
    frames_to_video(overlay_frames, "seq_00_overlay_thick.mp4", fps=20)
    print("Vidéo overlay (masque épaissi) écrite : seq_00_overlay_thick.mp4")

    print("Fin du masking_test")
