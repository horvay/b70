from glm_xpu import block_graphs as bg
for on in (False, True, False):
    bg.set_nosync(on)
    r, text, d = decode(24)
    print(f"{'no-sync' if on else 'exact'} {r:.2f} tok/s nvme={d.get('nvme')}: {text!r}", flush=True)
bg.set_nosync(False)
