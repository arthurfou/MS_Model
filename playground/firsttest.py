from ms_model.io.loaders import load_events



if __name__ == '__main__':
    print("Début du firsttest")

    ea = load_events("../../EVOwithMS/datasets/EV-IMO/eval/tabletop/npz/seq_00.npz")
    print(len(ea), ea.H, ea.W, ea.t.min(), ea.t.max(), ea.frame_ts.shape)

    print("Fin du firsttest")