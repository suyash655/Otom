import torch, sys
sys.path.insert(0, '.')

for ckpt_path in ['checkpoints/hybrid_best.pth', 'checkpoints/hybrid_dummy.pth']:
    try:
        c = torch.load(ckpt_path, map_location='cpu')
        sd = c.get('model_state_dict', c)
        tda_shape = sd['tda_norm.weight'].shape
        clf_shape = sd['classifier.0.weight'].shape
        num_out   = sd['classifier.3.weight'].shape[0]
        print(f"{ckpt_path}")
        print(f"  tda_norm.weight : {tda_shape}   (expected: [14])")
        print(f"  classifier.0    : {clf_shape}  (expected: [256, 526])")
        print(f"  num_classes out : {num_out}")
        print(f"  class_names     : {c.get('class_names', 'NOT SAVED')}")
        print()
    except Exception as e:
        print(f"{ckpt_path}: {e}\n")
