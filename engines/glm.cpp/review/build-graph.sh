#!/usr/bin/env bash
set -euo pipefail
cd ~/b70/engines/glm.cpp
CROOT=$HOME/opt/oneapi-2025.3/opt/intel/oneapi/compiler/2025.3
UROOT=$HOME/opt/oneapi-2025.3/opt/intel/oneapi/umf/1.0/lib
export LD_LIBRARY_PATH=$CROOT/lib:$UROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$CROOT/bin/icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg-g31' -O3 -std=c++20 review/graph-smoke.cpp -o review/graph-smoke -Lbuild-review -lglm -lglm_esimd -lglm_moe_pf -Wl,--disable-new-dtags,-rpath,$PWD/build-review:$CROOT/lib:$UROOT
