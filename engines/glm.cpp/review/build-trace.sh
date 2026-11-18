#!/usr/bin/env bash
set -euo pipefail
cd ~/b70/engines/glm.cpp
export CMPLR=$HOME/opt/oneapi-2025.3/opt/intel/oneapi/compiler/2025.3
export UMF=$HOME/opt/oneapi-2025.3/opt/intel/oneapi/umf/1.0/lib
export MKLROOT=$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3
export PATH=$CMPLR/bin:$PATH
export LD_LIBRARY_PATH=$CMPLR/lib:$UMF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
icpx -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -O3 -std=c++20 -ffast-math -fPIC -I src -c review/moe_trace.cpp -o review/moe_trace.o
./build.sh B=build-review ESIMD='build-review/exl3.o review/moe_trace.o' build-review/libglm_esimd.so
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg-g31' -O3 -std=c++20 -I src review/decode_trace.cpp -o review/decode_trace -Lbuild-review -lglm -lglm_esimd -lglm_moe_pf -L$MKLROOT/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -L$UMF -lumf -Wl,--disable-new-dtags,-rpath,$PWD/build-review:$CMPLR/lib:$UMF:$MKLROOT/lib
