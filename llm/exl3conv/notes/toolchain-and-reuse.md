# exl3conv: toolchain and reusable code

A survey, done 2026-10-23 without touching the GPU, of what a native EXL3 quantizer (C++ / SYCL / oneMKL / oneDNN,
no torch) can build with and reuse on this machine. Paths are absolute. "Verified" means I checked it here (file
listing, `nm`, or a CPU-only compile and link). "Recorded" means the claim comes from a doc in the repo, with the
doc named.

## 1. Summary

- **Compiler:** DPC++ 2025.3.2 at `~/opt/oneapi-2025.3`. You activate it by hand: set `PATH` and `LD_LIBRARY_PATH`,
  link `libumf` directly, and target `spir64_gen -device bmg-g31`. Every native build in the repo does it this way
  (template in section 3).
- **oneMKL 2025.3** (`~/opt/onemkl-2025.3-deb`) has the SYCL BLAS as a shared library (`gemm`, `syrk` on device). The
  **SYCL LAPACK (`potrf`/`potri`/`trtri`) exists only inside the 934 MB static `libmkl_sycl.a`**. A shared
  `libmkl_sycl_lapack.so.5` of the same release sits in the tts venvs, and I linked against it (section 2.2).
- **oneDNN 3.13.4** SYCL build `libdnnl_h3` (`~/opt/onednn-sycl-graph`): every SYCL binary here links it.
- **Reusable EXL3 GPU code** (section 5): `~/b70/llm/glm/xpu/csrc/exl3_esimd.h` is header-only ESIMD with no torch.
  It holds a bit-exact trellis reconstruct (W_hat to fp16), a Hadamard-128 in/out with suh/svh, a GEMV, and a DPAS
  GEMM, for the 3inst/mcg/mul1 codebooks. Its host launchers (`exl3_ops.sycl`) are torch-bound and have to be
  rewritten, which is small work. There is also a plain-SYCL sub-group Hadamard-128 in
  `~/b70/llm/dsv41/xpu/csrc/dsv_moe_glue.sycl`. **No Viterbi/encode kernel exists for XPU.** The only one is
  exllamav3's CUDA kernel (section 5.5).
- **CPU helpers:** nlohmann/json 3.11.2 single header, three safetensors readers (one is a standalone reader and
  writer), and bf16/fp16 bit conversions (section 8).
- **Running on the card:** always `b70-lease run <owner> -- <cmd>` or `b70q submit` (with `--chunked` for long work).
  The queue watches the kernel log for xe faults on the B70 (section 7).

## 2. Toolchain on disk

### 2.1 DPC++ / oneAPI 2025.3 (the one to use)

| Item | Path / value |
|---|---|
| Root | `~/opt/oneapi-2025.3/opt/intel/oneapi` (unpacked from Intel's apt debs, no root) |
| Compiler | `.../compiler/2025.3/bin/icpx`: `Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2 (2025.3.2.20260112)` (verified) |
| SYCL runtime | `.../compiler/2025.3/lib/libsycl.so.8`, UR loader 0.12, `libur_adapter_level_zero{,_v2}.so`, `libur_adapter_opencl.so` |
| UMF | `.../umf/1.0/lib/libumf.so.1`. It **must be linked directly** (`-Wl,--no-as-needed -lumf`), or SYCL sees no GPU (comment in `~/b70/engines/sd.cpp/build_sycl.sh`) |
| TBB / TCM | `.../tbb/2022.3`, `.../tcm/1.4` |
| setvars | `.../setvars.sh` exists, but **no build here uses it**. Every script sets `PATH`/`LD_LIBRARY_PATH` by hand, and MKL and oneDNN are separate trees that setvars doesn't know about. Don't use it. |
| AOT backend | `/usr/bin/ocloc` 26.31.39395 (from `intel-compute-runtime` 26.31.39395.13). icpx calls it for `spir64_gen`. `ocloc ids bmg-g31` gives IP 20.2.0 (Xe2) |
| GPU runtime | `/usr/lib/libze_loader.so.1.32.0` (level-zero-loader 1.32.0), `/usr/lib/libze_intel_gpu.so.1.17.39395`, IGC 2.40.13 |
| Level Zero headers | **not installed system-wide** (`/usr/include/level_zero/` has only `ze_intel_gpu.h` and friends, no `ze_api.h`). The repo uses `~/opt/onednn-src-h3/third_party/level_zero/level_zero/ze_api.h` (v1.11) through `ZE_INC` / `CPATH`. |
| OpenCL CPU runtime | `.../compiler/2025.3/lib/libintelocl.so` is present, but no ICD file registers it (`/etc/OpenCL/vendors/` has only `intel.icd`, the GPU, and `nvidia.icd`) |

### 2.2 oneMKL 2025.3 (`~/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3`)

`INTEL_MKL_VERSION 20250300`. `build_sycl.sh` calls it 2025.3.1, and its `libmkl_sycl_blas.so.5` is byte-identical
to the pip wheel `onemkl_sycl_blas-2025.3.1` (verified with `cmp`).

| Need | Status |
|---|---|
| Headers `oneapi/mkl/blas.hpp`, `oneapi/mkl/lapack.hpp` (+ `lapack/`) | present |
| SYCL BLAS on device (`column_major::gemm`, `syrk`, USM and buffer forms, fp32/fp16/bf16/int8) | `lib/libmkl_sycl_blas.so.5` (shared). Verified with `nm` |
| SYCL LAPACK on device (`lapack::potrf`, `potri`, `trtri`, `potrf_batch`, `*_scratchpad_size`) | **only in `lib/libmkl_sycl.a`** (static, 934 MB; symbols verified with `nm`). There is no `libmkl_sycl_lapack.so` in this tree. |
| Shared SYCL LAPACK, same release | `~/b70/tts/.venv-qwen/lib/libmkl_sycl_lapack.so.5` (pip `onemkl_sycl_lapack-2025.3.1`, NEEDED `libmkl_sycl_blas.so.5`, `libsycl.so.8`; RPATH `$ORIGIN`). The same file is in `tts/fish-speech/.venv-211/lib/`. |
| CPU LAPACK/BLAS (host `LAPACKE_spotrf` etc.) | `libmkl_core`, `libmkl_intel_{lp64,ilp64}`, `libmkl_{sequential,gnu_thread,intel_thread,tbb_thread}`, all shared and static |
| CMake / pkg-config | `lib/cmake/mkl/MKLConfig.cmake`; `lib/pkgconfig/mkl-*.pc` |

**Link probe (verified, CPU only, never run):** a file calling `blas::column_major::syrk`, `gemm`, `lapack::potrf`,
and the `potri`/`trtri` scratchpad sizes compiled and linked in about 5 s with:

```sh
icpx -O2 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  -DMKL_ILP64 -I$MKLROOT/include probe.cpp -o probe \
  -L$MKLROOT/lib -lmkl_sycl_blas -L$HOME/b70/tts/.venv-qwen/lib -l:libmkl_sycl_lapack.so.5 \
  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -lpthread -lm -ldl \
  -L$UMF -Wl,--no-as-needed -lumf -Wl,--as-needed -Wl,-rpath,$CMPLR/lib:$UMF:$MKLROOT/lib:$HOME/b70/tts/.venv-qwen/lib
```

Without `-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` the link fails with undefined `cblas_*_64` and `mkl_serv_*`.
The SYCL interfaces are ILP64 (`std::int64_t` sizes), so build with `-DMKL_ILP64`.

LAPACK options, in order of preference:

1. **Shared `.so.5` from the venv** (works now, verified link). It is fragile, though: deleting a tts venv breaks
   the binary. Copying that one file next to `libmkl_sycl_blas.so.5` in `~/opt/onemkl-2025.3-deb/.../lib`, or
   unpacking Intel's `intel-oneapi-mkl-sycl-lapack-2025.3` deb the way the other MKL debs were unpacked, would make
   it permanent. That is Greg's call. I installed nothing.
2. **Static `libmkl_sycl.a`.** Untested. The device code of everything pulled in has to be linked, and with
   `spir64_gen` AOT-compiled too, so expect long links and high RAM. Try it only if 1 is refused.
3. **Host MKL LAPACK** (`LAPACKE_spotrf` / `spotri` on the CPU, then upload). A 16384^2 fp32 Cholesky is about
   1.5 TFLOP, a few seconds on the 16-core Ryzen. That is fine for one Hessian per linear and needs no device LAPACK.

The first call of an MKL SYCL routine may JIT-compile kernels for the device. Set `SYCL_CACHE_PERSISTENT=1`.

### 2.3 oneDNN

| Build | Path | Notes |
|---|---|---|
| **libdnnl_h3** (use this) | `~/opt/onednn-sycl-graph/{include,lib/libdnnl_h3.so.3.13}` | v3.13.4 + local SDPA patch, `ONEDNN_GPU_RUNTIME=SYCL`, `ONEDNN_CPU_RUNTIME=NONE`, `ONEDNN_ENABLE_PRIMITIVE_GPU_ISA=XE2`, Graph API on, `ONEDNN_ENABLE_WORKLOAD=INFERENCE` (no backward primitives). Built by `~/b70/engines/onednn/build.sh` from `~/opt/onednn-src-h3`. Link with `-ldnnl_h3`. |
| libdnnl (older) | `~/opt/onednn-sycl/lib/libdnnl.so.3.13` | same version, no Graph API. Used only by the deprecated torch builds (`xpu_sdpa/build_icpx.sh`). |

The rule is one oneDNN per process. Anything that shares a process with sd-cli or the xpu libraries must link
`dnnl_h3`. oneDNN has no CPU engine here (`CPU_RUNTIME=NONE`), so the CPU fallback is MKL.

### 2.4 oneAPI 2026.0 in the SGLang image (don't mix it in)

`~/b70/llm/state/image/rootfs/opt/intel/oneapi/` (and a copy under `llm/trellis/image/rootfs/`) holds
compiler/MKL 2026.0: `icpx`, `libsycl.so.9`, `libmkl_sycl_lapack.so.6` (NEEDED `libsycl.so.9`), and dnnl. It is
**ABI-incompatible with 2025.3** (`libsycl.so.8`). The torch-XPU servers use it inside bubblewrap. A native binary
should stay on 2025.3 so that it shares a SYCL runtime with sd-cli, the xpu libraries, and drama3_batch. Building
against 2026.0 would mean compiling and running inside the image.

## 3. How existing native SYCL builds are configured

Sources: `~/b70/engines/sd.cpp/build_sycl.sh`, `~/b70/engines/sd.cpp/xpu_sdpa/build_native.sh`,
`build_music3.sh`, `~/b70/engines/audio.cpp/build_sycl.sh`, `~/b70/engines/onednn/build.sh`,
and `engines/sd.cpp/build-sycl/CMakeCache.txt`.

CMakeCache (sd-cli): `CMAKE_CXX_COMPILER=.../icpx`, `CMAKE_CXX_FLAGS=-Wno-narrowing -fsycl`, `-O3 -DNDEBUG`,
`GGML_SYCL_TARGET=INTEL`, `GGML_SYCL_DEVICE_ARCH=bmg-g31`, `MKL_INTERFACE=ilp64`, `MKL_THREADING=sequential`,
`MKL_LINK=dynamic`, and the linker line
`-static-intel -Wl,--disable-new-dtags -L$UMF -Wl,--no-as-needed -lumf -Wl,--as-needed -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31"`.

The standalone native pattern (`build_native.sh`, `build_music3.sh`; no CMake, one `icpx` call per artifact)
adapted for exl3conv:

```sh
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
MKLROOT=${MKLROOT:-$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3}
MKL_LAPACK=${MKL_LAPACK:-$HOME/b70/tts/.venv-qwen/lib}      # see 2.2
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF:$MKLROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nice -n 10 icpx -O2 -std=c++17 -static-intel \
  -fsycl -fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  -fp-model=precise -fno-sycl-id-queries-fit-in-int -DMKL_ILP64 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
  -I"$MKLROOT/include" -I"$ONEDNN/include" -I"$HERE" -I"$HOME/b70/engines/sd.cpp/thirdparty" \
  "$HERE/exl3conv.cpp" -o "$HERE/exl3conv.new" \
  -L"$MKLROOT/lib" -lmkl_sycl_blas -L"$MKL_LAPACK" -l:libmkl_sycl_lapack.so.5 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
  -L"$ONEDNN/lib" -ldnnl_h3 -L"$UMF" -Wl,--no-as-needed -lumf -Wl,--as-needed \
  -Wl,-rpath,"$CMPLR/lib:$UMF:$MKLROOT/lib:$MKL_LAPACK:$ONEDNN/lib"
mv "$HERE/exl3conv.new" "$HERE/exl3conv"      # rename, so a running process keeps its mapped copy
ldd "$HERE/exl3conv" | grep -Ei 'vulkan|torch|c10' && { echo "unexpected Vulkan/torch dependency" >&2; exit 1; } || true
ldd "$HERE/exl3conv" | grep "not found" && { echo "missing shared libraries" >&2; exit 1; } || true
```

What each flag is for, according to the scripts' own comments:

- `-fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31"`: AOT for the B70 (no IGC
  JIT at startup; the JIT took ~60 s for GLM's kernels and "minutes" for DSV4.1's), plus a SPIR-V copy for JIT
  elsewhere. `-Xs "-device bmg-g31"` is the short form (`llm/dsv41/xpu/build.sh`). When a static library holds the
  kernels, **the executable's link line must name the target**, or the run fails with "No kernel named ... was
  found" (`build_sycl.sh`).
- `-fp-model=precise`: icpx defaults to fast floating point. The xpu libraries need IEEE infinities and the exact
  op order. Use it for the Viterbi costs and for anything compared bit for bit. (The trellis-serve and GLM EXL3
  builds use `-O3 -ffast-math` and are still bit-exact, because the codebook decode is integer ops plus one fp16
  FMA.)
- `-fno-sycl-id-queries-fit-in-int`: needed when global ranges or ids can pass 2^31.
- `-static-intel`: links Intel's libsvml/libimf/libintlc in. `-Wl,--disable-new-dtags` (CMake builds) makes RPATH
  also cover libraries that get dlopened.
- `-fsycl-device-code-split=per_kernel`: used by the torch-op builds (`llm/glm/xpu/build.sh`, `llm/dsv41/xpu/build.sh`).
- Parallelism: `nice -n 10`, `-j ${JOBS:-3}`, one compile at a time. Leave `-fsycl-max-parallel-link-jobs` at its
  default of 1. (ggml's CMake has `GGML_SYCL_MAX_PARALLEL_LINK_JOBS=32`, which is not what we want on a shared
  machine.)
- **ESIMD + AOT:** `llm/dsv41/xpu/build.sh` reports that "ocloc rejects its AOT link ('more than one module with an
  entry point')" for the ESIMD MoE kernel built with `-fsycl-device-code-split=per_kernel`, and falls back to
  `spir64` JIT. The GLM and trellis-serve EXL3 ESIMD libraries are JIT (`spir64`) too. On the other hand, ggml-sycl's
  ESIMD DPAS (`engines/audio.cpp/external/ggml/src/ggml-sycl/mmxmx.cpp`) is AOT-built for `bmg-g31` without trouble.
  Plan: put ESIMD kernels in their own translation unit, try AOT, and fall back to JIT with
  `SYCL_CACHE_PERSISTENT=1` if ocloc complains.

## 4. Runtime environment (picking the B70)

| Variable | Value | Why / where |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` | "SYCL sees only the B70 (Level Zero device 0; the 3080 Ti has no Level Zero driver)" (`tools/b70-image.sh`, `tools/h3-ref2va`, `tools/drama3_clone`). It also hides the OpenCL platforms: `nvidia.icd` is registered system-wide. |
| `ZE_AFFINITY_MASK` | `0` | set by `llm/b70` for every server |
| `CUDA_VISIBLE_DEVICES` | `""` | `tools/drama3_clone` sets it, as a belt-and-braces measure |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | allows single device allocations above 4 GB. It has to be set before the first platform query; `xn::rt()` in `xpu_native.hpp` calls `setenv(..., 0)` |
| `SYCL_CACHE_PERSISTENT` / `SYCL_CACHE_DIR` | `1` / a cache dir | keeps JIT results across runs (`llm/b70`) |
| `NEOReadDebugKeys=1 TreatNonUsmForTransfersAsSharedSystem=0 EnableSharedSystemUsmSupport=0` | | the Gemma server sets these: "the copy engine reading a swapped page in place faulted the card while loading" (`llm/b70`, `serve_gemma`) |

In code, `xn::pick_device()` (`~/b70/engines/sd.cpp/xpu_sdpa/xpu_native.hpp:421`) takes the first GPU
on the `ext_oneapi_level_zero` backend, and falls back to `gpu_selector_v`. Use one in-order queue.

The B70 is PCI `0000:0a:00.0` (`8086:e223`, "Battlemage G31 [Arc Pro B70]") on the `xe` driver, currently
`/sys/class/drm/card2`; the NVIDIA card is `0000:04:00.0`. DRM card numbers can change, so match on the PCI
address.

## 5. Existing EXL3 GPU code

### 5.1 Inventory

| File | Kind | Ops | torch-free? |
|---|---|---|---|
| `~/b70/llm/glm/xpu/csrc/exl3_esimd.h` (938 lines; identical to `llm/trellis/trellis-serve/xpu/csrc/exl3_esimd.h`, trellis-serve `72fd2d4`) | ESIMD header | codebook decode `decode_cb_h<CB>` (CB 0 = 3inst, 1 = mcg, 2 = mul1, bit-exact with exllamav3), `LutKernel` (65536-entry fp16 LUT), `fwht128` (in-register Sylvester FWHT, unnormalised), `HadInKernel` (xh = fp16(H(fp16(x*suh))/sqrt128), blocked or row-major out), `HadOutKernel` (out = svh * H(sum_p part)/sqrt128), int8 variants `HadInQ8Kernel`/`HadInQ8WgKernel`/`HadOutQ8Kernel`, `GemvKernel<K,CB,MR,NT>` (decode-in-GEMV, M <= 4), `DpasKernel<K,CB,MB,NT>` (XMX, M 5..64), **`ReconstructKernel<K,CB,NT,TW>`** (trellis to W_hat fp16 or int8, bit-exact) | **yes**: includes only `sycl.hpp` + ESIMD headers |
| `~/b70/llm/glm/xpu/csrc/exl3_ops.sycl` | host launchers + `TORCH_LIBRARY(exl3xpu_C)` | `exl3_linear` (small M fused, large M: had_in_rm, then reconstruct slices, then oneDNN/`at::matmul`, then had_out), `exl3_reconstruct`, `exl3_had_in_rm`, `exl3_had_out_h`, `exl3_gemm_small`, `exl3_gemm_raw`, oneDNN int8 GEMM, oneDNN Graph SDPA, ESIMD flash attention | **no**: `at::Tensor`, `c10::xpu` queue, `TORCH_CHECK`. The launchers are thin (`launch_reconstruct` is 10 lines) and easy to port to raw pointers. |
| `~/b70/llm/glm/xpu/csrc/fa_esimd.h` | ESIMD header | prefill flash attention, fp16, head_dim 256 | yes (not needed) |
| `~/b70/llm/glm/xpu/csrc/exl3_moe{,.n128}.sycl` | ESIMD + torch | grouped MoE GEMV/DPAS over pointer tables (GLM blob format) | no |
| `~/b70/llm/dsv41/xpu/csrc/dsv_moe.sycl` | ESIMD + torch | mixed-K (1..4) grouped EXL3 GEMV on `exl3_esimd.h`'s `GemvKernel` | no |
| `~/b70/llm/dsv41/xpu/csrc/dsv_moe_glue.sycl` | plain SYCL + torch launchers | **`fwht(sycl::sub_group&, float v[8])`**: a 128-block held by one 16-lane sub-group, 8 values per lane, register butterflies for strides 1/2/4 and `permute_group_by_xor` for 8..64; MoE prep/mid/post that apply suh/svh | device code yes, launchers no |
| `~/b70/llm/glm/xpu/exl3xpu/triton_kernels.py`, `~/b70/llm/dsv41/xpu/dsv_xpu/exl3_tri.py` | Triton (torch) | reconstruct / dequant (any integer K 1..8, mul1), had_in, had_out, gemm_small | no (good as a readable spec) |
| `~/b70/engines/sd.cpp/xpu_sdpa/xpu_native.hpp` | plain SYCL + oneDNN, no torch | `rotate_groups` Hadamard: **powers of 4 only (4..4096), so not 128**; oneDNN `matmul()` (f32/bf16/f16/s8 with scales, cached primitives); USM `Pool`; pinned `Staging` (64 MiB chunks); safetensors reader | **yes** (needs `dnnl_h3`) |
| `~/b70/llm/trellis-serve.patch` | patch to trellis-serve | cold_read.c, MoE/n-gram/staging changes for Gemma and Flash-Next | nothing for the quantizer |

The other half of the GLM build (`llm/glm/xpu/build.sh`, run inside the image) produces
`llm/glm/xpu/exl3xpu/_C.so` and `_moe_glm.so`. Those are torch extensions and cannot be used from a native binary.

### 5.2 Coverage limits of `exl3_esimd.h`

- `Geo<K>` has `static_assert(V >= 8)` with `V = 32/gcd(K,32)`: **K = 8 does not compile**. K = 1..7 fit the
  template.
- `exl3_reconstruct` in `exl3_ops.sycl` only instantiates (K, cb) = (3,2), (4,2), (5,2), (6,2), plus (2,2), (4,1),
  (4,0) under `-DEXL3_ALL_CODEBOOKS`. Other K need new instantiations, which nobody has tested yet.
- **Half-integer K** (the fractional rates of exllamav3 1.6, `FRAC_MASK 0xAAAA`, CUDA comp units `h1/h2/h3`) are not
  handled by any XPU kernel.
- `ReconstructKernel` launch: `NT = 8`, `n_out % 128 == 0`, `n0 % 128 == 0`,
  `nd_range(round_up(n_strips*tk, 8), 8)` with `n_strips = n_out/128` and `tk = in/16`
  (`exl3_ops.sycl:324-333`).

### 5.3 Dequantizing an EXL3 tensor to fp16 (for checking packed output)

No single kernel exists. Two ways to build one from the pieces above:

- **Bit-exact path (exllamav3's unfused order):** `ReconstructKernel` gives W_hat `[in, out]`. Then apply
  `W = diag(suh) Hblk W_hat Hblk diag(svh)`. Row-wise over the out dimension, `HadOutKernel<fp16, fp16>` with `P = 1`
  computes `svh * H(row)/sqrt128`. Transpose to `[out, in]` (the HF weight layout) and run `HadOutKernel` again,
  with `suh` passed as the scale. Exact rounding rules: `~/b70/llm/glm/docs/ext-op-inventory.md`
  §1.2.3 to 1.2.6 (had_r_128 fp16 path: pre-scale `__hmul2` in fp16, butterfly fp32, `*r` fp32, round to fp16, then
  post-scale in fp16).
- **Reference on CPU** (no GPU needed to validate a packer): the numpy reference
  `~/b70/llm/dsv41/exllamav3/tests/testlib/trellis.py` (`unpack`, `pack`, `decode`, `dequant_rotated`,
  `dequant`; handles integer and half-integer K, all three codebooks), the torch reference
  `~/b70/llm/trellis/trellis-serve/core/src/trellis_core/reference.py`, and the format spec in
  `ext-op-inventory.md` §1.3. A C++ port of `trellis.py` can serve as a CPU check of the packer.

### 5.4 Calibration forward through quantized layers

For calibration batches (many rows), the cheapest route is the GLM prefill path rebuilt without torch:
`HadInKernel` (row-major, `x*suh`, fp16), `ReconstructKernel` in column slices (fp16 W_hat), the oneDNN fp16 GEMM
(`xn::matmul` in `xpu_native.hpp`, ~140 TFLOPS recorded for this shape in trellis-serve `PROGRESS.md`), then
`HadOutKernel` with `svh`. The small-M `GemvKernel`/`DpasKernel` launchers carry a lot of tuning state
(`exl3_ops.sycl:81-320`) and are not worth porting for calibration.

### 5.5 Viterbi / trellis encode

**None on XPU, in SYCL or Triton.** The only implementation is exllamav3's CUDA, in both submodules:

- `~/b70/llm/glm/exllamav3/exllamav3/exllamav3_ext/quant/` (exllamav3 1.5.1, horvay `b70-xpu`) and
  `~/b70/llm/dsv41/exllamav3/exllamav3/exllamav3_ext/quant/` (1.6.1.post1, vcruz305; has half-integer K):
  `quantize_tiles_kernel.cuh` (templated Viterbi, ring length L = 256 or 160), `quantize_tiles_optimized.cuh`,
  `quantize_tiles_frac_kernel.cuh`, `quantize.cu` (launchers: K 1..8, fp16 `temp_costs`, byte/short `temp_edges`),
  `pack.cu` (`pack_trellis`: 16-bit MSB-first, then SWAP16 per uint32), `codebook.cuh`, `reconstruct.cu`,
  `hadamard*.cu`, `comp_units/quantize_tiles_inst_k{1..8}.cu`.
- Python driver: `.../exllamav3/modules/quant/exl3_lib/quantize.py` (1966 / 2201 lines); conversion loop:
  `.../exllamav3/conversion/`.
- Tests worth porting: `~/b70/llm/dsv41/exllamav3/tests/kernels/quant/test_quantize_tiles.py`,
  `test_quantize_tiles_frac.py`.

## 6. B70 hardware facts recorded in the repo

| Fact | Value | Source |
|---|---|---|
| Arch / AOT name | Xe2, `bmg-g31`, IP 20.2.0 | `ocloc ids`, build scripts |
| Xe-cores | 32 (`gpu_subslice_count`) | `llm/glm/docs/ext-op-inventory.md:1818` |
| Clock | max 2800 MHz (sysfs); seen PL2-throttled to 2600 during decode at the 230 W cap | sysfs; `llm/trellis/trellis-serve/xpu/docs/PROGRESS.md:102` (0xSero's B70) |
| Sub-group size | every plain-SYCL kernel here uses `[[sycl::reqd_sub_group_size(16)]]` (dsv41 csrc, xpu_sdpa). DPAS execution size on Xe2 = 16 | grep; `mmxmx.cpp` comment |
| DPAS shapes | fp16: B operand K16 x N16 in VNNI (`((k>>1)*16+n)*2+(k&1)`), repeat count 8 rows; int8: K = 32. One EXL3 16x16 tile is exactly one DPAS B operand | `exl3_esimd.h:583-590`, `mmxmx.cpp` |
| GRF | 128 registers default; `grf_size<256>` (large GRF, 16 KB per thread) for big accumulators. **In large-GRF mode an Xe-core holds 32 hardware threads: a 64-thread work-group could not be resident and hung the B70** | `exl3_moe.n128.sycl:880-886`, `llm/glm/docs/dev-notes.md:40`, `PROGRESS.md:145` |
| SLM | no measured or queried value is recorded. exllamav3's Triton picker was given "a fixed 64 KB SLM budget" on XPU. Query `sycl::info::device::local_mem_size` at startup (on the card, under the queue) rather than hardcoding | `llm/dsv41/README.md:100` |
| VRAM bandwidth | EXL3 GEMV DRAM ceiling ~531 GB/s measured (decode stripped); EXL3 weight streaming 415-630 GB/s; FP8 GEMV 550-590 GB/s; DSV4.1 expert GEMVs from VRAM 181 (K=1) to 448 (K=4) GB/s; GLM ~450 GB/s | `PROGRESS.md:53`, `DESIGN.md:36`, `llm/dsv41/README.md:91,211`, `llm/glm/README.md:54` |
| PCIe | 4.0 x16 behind its own switch, ~24-27 GB/s H2D; staging copies 28 GB/s | `llm/glm/docs/dev-notes.md:94`, `llm/trellis/README.md:34` |
| Compute | oneDNN bf16 GEMM ~100-150 TFLOPS; fp16 prefill GEMM ~140; int8 XMX 2x the fp16 rate; oneDNN SDPA ~95; vector FMA/DP4A ~1.35 T multiply-adds/s (Vulkan measurement) | `engines/sd.cpp/docs/b70_xpu.md`, trellis-serve `xpu/README.md:54`, `bench/drama3/OPTIMIZE.md:74` |
| Host | Ryzen 9 5900XT (16C/32T), 62 GiB RAM shared with the desktop | `llm/glm/docs/dev-notes.md:95` |
| VRAM | 32 GB (31.9 GiB reported by `b70q`); past it the driver moves buffers to system memory and the job crawls | `xpu_native.hpp` header |

### 6.1 Driver pitfalls and how existing code avoids them

1. **5 s job timeout.** sysfs `.../tile0/gt0/engines/{ccs,rcs,bcs}/job_timeout_ms = 5000` (max 10000). Stock sd.cpp's
   Vulkan flash attention "outlived the xe driver's 5 s job timeout" and lost the device (`README.md`); a DSV4.1
   variant "took the GPU past the 10 s timeout". **Keep every launch well under a second.** A Viterbi pass over a
   big tensor must be split across many launches (batches of tiles).
2. **Large host-to-device copies fault the copy engine** ("Engine memory CAT error, class=bcs", engine reset, job
   hangs; intel/compute-runtime#1010) when a large pinned/userptr source is under host memory pressure. Fixes in use:
   copies in pieces of at most 32 MiB (`tts/README.md:249`), two bounded 64 MiB pinned staging buffers
   (`xn::Staging`), and the NEO debug keys in section 4.
3. **Device atomics on host USM** fault: xe can only service an atomic page fault by migrating the page, and a host
   buffer can't move ("Engine memory CAT error", AccessType 2). Use device memory for atomics, and plain cache-bypass
   stores for host-visible flags (`llm/trellis/README.md:168-180`, `llm/flashnext/README.md:62`).
4. **Large-GRF work-groups above 32 threads hang** (see table above). The GLM MoE splits work-groups to <= 32.
5. **Allocations:** one pinned allocation above ~4 GB silently returns null; O_DIRECT into USM host memory gives
   EFAULT; single device allocations above 4 GB need `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
   (`llm/glm/docs/dev-notes.md:97`, `xpu_native.hpp:436`). An out-of-memory condition can show up as a Level Zero
   segfault instead of an error (`dev-notes.md:54`). Large conv batches gave `UR_RESULT_ERROR_OUT_OF_RESOURCES`
   while most VRAM was free (`tts/README.md`).
6. **Faulting kernels that keep running have wedged the driver**, and once froze the machine
   (`llm/trellis/README.md:129`, `llm/README.md`). That is why the guards stop a process at the first fault.
7. **A Steam game under Proton opens the B70 too.** Model loads beside one failed with OUT_OF_DEVICE_MEMORY /
   DEVICE_LOST (`llm/b70` Guard). `b70 serve` refuses to start next to one.
8. A failed XPU graph capture left later replays faulting (torch-specific, `llm/glm/README.md:122`). It doesn't apply
   to plain SYCL, but don't replay command graphs after a failed record.

## 7. Running a native GPU binary here

From `~/b70/queue/README.md` and `queue/src`:

- A job you wait on: `b70-lease run exl3conv --note "layer 12" -- ./exl3conv ...` (blocks and returns the exit code;
  `B70Q_TICKET` is set inside, and nested `b70-lease run` calls reuse it).
- Long work: `b70q submit exl3conv --chunked --retries 1 --log /abs/log -- ./exl3conv ... --resume`. The command does
  about 20-30 minutes of work, then **exits 99 if more remains** and 0 when done; each chunk goes back in line. This
  fits a quantizer that checkpoints per layer. Use `b70q wait <id>` and `b70q log <id> -f`.
- Holding the card across tool calls: `b70-lease acquire exl3conv --ttl 1h`, then `export B70Q_TICKET=<token>`, then
  `b70-lease release <token>`.
- `b70q` prints status (just now, `#233 make_real` held the card at 24.7/31.9 GiB). Turns are 30 minutes, first come
  first served, and nothing is preempted. `b70q pause [reason]` / `b70q resume` stop and restart granting.
  `b70q now -- <cmd>` puts your own run next in line.
- **Fault guards:** the daemon follows `journalctl -k -f` for lines containing `xe <pci-slot>` that match
  `CAT error|Timedout job|wedged|GT reset|coredump has been created|Fault response: Unsuccessful`
  (`queue/src/kernel.ts`). On a match it raises an alert and holds the queue for 2 minutes. **Three faults in an
  hour pause the queue until someone runs `b70q resume`** (`queue/src/queue.ts:629-637`, `config.ts:39-40`).
  `b70 serve` adds its own `Guard` (`llm/b70:619`, regex `Fault|CAT error|[Hh]ang|[Rr]eset|wedged|Timedout` on
  `xe 0000:0a:00.0` lines), which stops the process group. `tts/b70_load.py`'s `xe_watchdog` does the same for the
  torch TTS jobs. exl3conv should do the same: follow `journalctl -k` from its start time and exit at the first
  B70 fault, so a faulted job fails fast instead of hanging.
- Loading SYCL/Level Zero libraries at all counts as using the card (`AGENTS.md`). Even a SYCL CPU-device run (the
  unregistered `libintelocl.so`, through `OCL_ICD_FILENAMES` + `ONEAPI_DEVICE_SELECTOR=opencl:cpu`; untested) should
  go through the queue, or be cleared with Greg first.

## 8. CPU-side helpers worth reusing

| Need | Where | Notes |
|---|---|---|
| JSON (config.json, quantization_config, shard index) | `~/b70/engines/sd.cpp/thirdparty/json.hpp` (nlohmann 3.11.2, MIT, single header) | add `-I.../thirdparty` |
| safetensors read + write, standalone | `~/b70/engines/audio.cpp/include/engine/framework/io/safetensors.h` + `src/framework/io/safetensors.cpp` (488 lines, std only; own small JSON scanner with a nesting limit, `__metadata__` map) | Apache-2.0 (ShugoAI). `encode_safetensors` builds the whole file in memory, so for multi-GB shards write a streaming writer (header first, then tensors) |
| safetensors read (+ index.json shards, streaming writer) | `~/b70/engines/sd.cpp/src/model_io/safetensors_io.{h,cpp}` | MIT, but tied to ggml's `TensorStorage` / `ggml_type`. Use it as reference code, not as a library |
| safetensors read, minimal | `xn::SafeTensors` in `~/b70/engines/sd.cpp/xpu_sdpa/xpu_native.hpp:554-669` | std::regex over the header, assumes key order `dtype, shape, data_offsets` (true for files from the Rust `safetensors` library), no shard index; `host_f32()` converts F32/BF16/F16/I8/U8; `upload_raw()` streams into device memory through pinned staging. Slow on huge headers |
| bf16 / fp16 bits | `xn::f32_to_bf16_bits` (RNE, NaN-safe), `xn::bf16_bits_to_f32`, `xn::f16_bits_to_f32` (subnormals handled) at `xpu_native.hpp:111-153` | no host f32 to f16 function there; `sycl::half` and `_Float16` work in icpx host code |
| Device dtype casts, oneDNN matmul, USM pool | `xn::convert_any`, `xn::matmul(MatmulArgs)`, `xn::Pool`, `xn::Buf` (`xpu_native.hpp`) | header-only, links `dnnl_h3` |
| O_DIRECT parallel reader | `~/b70/llm/dsv41/xpu/csrc/dsv_io.c`, trellis-serve `xpu/csrc/cold_read.c` | for streaming big shards at ~5.5-6.5 GB/s |
| Self-test pattern | `~/b70/engines/sd.cpp/xpu_sdpa/SELFTEST.md`, `xpu_selftest.cpp`, `selftest.sh`: a CPU-only `gen` step writes inputs, a reference step writes `ref_out.f32`, a `check` step runs the native code and prints RMS / relative RMS / max abs error, failing above `*_TOL` | `xn::compare_report` at `xpu_native.hpp:1269` |

## 9. Conventions for new native code

- **Where:** LLM-side tools live under `llm/<name>/` with `README.md`, sources, and `build.sh` next to them (e.g.
  `llm/dsv41/xpu/{csrc,build.sh}`). Shared engines are submodules under `engines/` (commit there, push only when Greg
  asks). Top-level commands go in `tools/`. `llm/exl3conv/` fits the pattern. Models stay in `~/models`
  (`tools/mstore.ts`); never commit weights. Put build outputs in `.gitignore` (`llm/glm/.gitignore` ignores `*.so`).
- **Build scripts:** bash with `set -euo pipefail`; a header comment listing the outputs, environment overrides
  (`ONEAPI`, `MKLROOT`, `ONEDNN` with `~/opt` defaults) and usage; `nice -n 10`; `JOBS` default 3; write `X.new`,
  then `mv` it into place; finish with `ldd` checks for `vulkan|torch|c10` and "not found". The native code uses
  plain `icpx` calls, while CMake is used where upstream ggml already has it.
- **Source headers:** each file opens with a `//` block saying what it is, its layout and formulas, its environment
  variables with defaults, and measured numbers with dates (see `xpu_native.hpp`, `exl3_esimd.h`, `dsv_moe.sycl`).
- **License:** the repo is MIT (`~/b70/LICENSE`, `~/b70/llm/LICENSE`, (c) 2026 Greg Horvay).
  Derived work is credited in a `NOTICE` (`~/b70/llm/NOTICE`, `~/b70/llm/glm/NOTICE`). A port of
  exllamav3's quantizer must credit **ExLlamaV3 (MIT, (c) 2025 turboderp)**; reused `exl3_esimd.h` code must credit
  **trellis-serve (MIT, (c) 2026 0xSero)** with its commit (`llm/glm/xpu/TRELLIS_COMMIT` = `72fd2d4c...`); copying
  audio.cpp's safetensors code brings in **Apache-2.0** terms.
- **README style:** plain prose and tables, measured numbers with the setup that produced them, quality checks next
  to speed claims (top-1 / KL), and a list of what didn't work and why (see `llm/glm/README.md`,
  `bench/drama3/OPTIMIZE.md`).

## 10. Open items

- Pick the LAPACK route (2.2): the venv `.so.5` works now, but Greg should approve copying it into `~/opt` or
  unpacking the deb.
- `local_mem_size`, `max_work_group_size`, `sub_group_sizes` and `max_mem_alloc_size` of the B70 are not recorded
  anywhere. Have a tiny probe print them in the first queued GPU run.
- K = 8 and half-integer K have no XPU decode: they need a generalized reconstruct kernel. The `trellis.py` numpy
  reference covers both, for testing.
