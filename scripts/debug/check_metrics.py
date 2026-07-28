import json
m = json.load(open('checkpoints/metrics.json'))
h = m.get('history', [])
print(f'Epochs completed: {len(h)}')
for e in h:
    ep = e["epoch"]
    ta = e["train_acc"]
    va = e["val_acc"]
    vf = e["val_macro_f1"]
    print(f'  Ep {ep:>2}: train_acc={ta:.3f}  val_acc={va:.3f}  val_F1={vf:.3f}')
print(f'Best val_F1: {m["best_val_macro_f1"]:.4f}')
