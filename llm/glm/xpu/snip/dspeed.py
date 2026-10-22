# Decode-only rate: difference of an 8-token and a 40-token run on the same prompt
import time
r = {}
for n in (8, 40):
    t, _, d = decode(n, warm=False)
    r[n] = (n / t, d)
t8, t40 = 8 / r[8][0], 40 / r[40][0]
print(f"decode-only {32 / (t40 - t8):.2f} tok/s  (8 tok {t8:.1f} s, 40 tok {t40:.1f} s)")
d8, d40 = r[8][1], r[40][1]
per = {k: round((d40[k] - d8.get(k, 0)) / 32, 2) for k in d40}
print("per decode token:", per)
