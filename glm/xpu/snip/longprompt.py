exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
run(True, 8)
print("--- 3.6K prompt, then decode", flush=True)
N = 3600
exec(open("/w/xpu/snip/prefill.py").read())
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
run(True, 32)
