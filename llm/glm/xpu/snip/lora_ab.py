# LoRA A/B (devserver): the same GM prompts and seeds with the adapter off and on. The adapter is applied before the
# first decode (the decode graphs capture it) and switched with set_scale, in place. Writes /w/run/lora_ab.txt.
import torch
from glm_xpu import lora
from exllamav3.generator.sampler import ComboSampler

SYS = open("/w/xpu/data/gm-system-prompt.txt").read().strip()
PROMPTS = [
    "I push open the door of the roadside tavern, shake the rain off my cloak and look around.",
    "I draw my sword and tell the bandit leader he has one chance to let the merchant go.",
    "I sneak into the wizard's tower at night and start searching his desk for the stolen map.",
]
import json
if "LORA_AD" not in globals():
    LORA_AD = lora.apply(model, "/lora", 1.0)
AD = LORA_AD
EOS = json.load(open("/models/generation_config.json"))["eos_token_id"]
out = []
for i, p in enumerate(PROMPTS):
    prompt = model.default_chat_prompt(p, SYS) + "</think>"
    for tag in ("off", "on"):
        lora.set_scale(AD, 0.0 if tag == "off" else 1.0)
        gen.pagetable.reset_page_table()               # no prefix reuse across the off / on runs
        torch.manual_seed(1000 + i)
        t = gen.generate(prompt=prompt, max_new_tokens=400, completion_only=True, seed=1000 + i,
                         sampler=ComboSampler(temperature=0.8, top_p=0.95), stop_conditions=EOS)
        out.append(f"=== prompt {i} LoRA {tag} ===\n{p}\n---\n{t.strip()}\n")
        print(out[-1][:200].replace("\n", " "), "...", flush=True)
open("/w/run/lora_ab.txt", "w").write("\n".join(out))
print("wrote /w/run/lora_ab.txt")
