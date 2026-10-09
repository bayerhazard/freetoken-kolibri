"""Register Kolibri1ForCausalLM in the installed freetoken registry.

Runs at image build time. Locates register.py via sysconfig (no freetoken import,
so no torch/CUDA init during build). Idempotent and corrective: any previous
Kolibri entry is removed before the correct one is inserted.
"""

import os
import re
import sys
import sysconfig

REGISTER = os.path.join(
    sysconfig.get_paths()["purelib"], "freetoken", "models", "register.py"
)
ENTRY = (
    '    "Kolibri1ForCausalLM": ModelSpec(\n'
    '        "freetoken.models.kolibri",\n'
    '        "Kolibri1ForCausalLM",\n'
    '        packed_modules_mapping=(("qkv_proj", ("q_proj", "k_proj", "v_proj")),),\n'
    "    ),\n"
)

if not os.path.exists(REGISTER):
    raise SystemExit(f"not found: {REGISTER}")

with open(REGISTER) as f:
    src = f.read()

# Drop any earlier Kolibri entry (from an older build).
src = re.sub(
    r'    "Kolibri1ForCausalLM": ModelSpec\(.*?\n    \),\n', "", src, flags=re.S
)

anchor = "_MODEL_REGISTRY: dict[str, ModelSpec] = {\n"
if anchor not in src:
    raise SystemExit(f"anchor not found in {REGISTER}")

src = src.replace(anchor, anchor + ENTRY, 1)
with open(REGISTER, "w") as f:
    f.write(src)
print("patched", REGISTER)
