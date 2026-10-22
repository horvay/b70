# Agent notes

- **One home for B70 code.** New B70 engines, scripts and servers go here, not in project folders or new checkouts;
  projects call `~/b70/tools/*`, `~/b70/engines/sd.cpp/build-sycl/bin/sd-cli` and `~/b70/engines/sd.cpp/xpu_sdpa/*`.
  One sd.cpp checkout and one sd-cli build serve every job; add features to it instead of making a branch worktree.
- **Every GPU job goes through the queue** (`queue/README.md`): `b70-lease run`, `b70q submit`, `b70q serve --idle`.
  Loading SYCL/Level Zero/XPU libraries counts as using the card. No own locks, sleep gaps, `pgrep` chains or
  killing other jobs by process name.
- **Intel stack only:** SYCL, oneDNN, oneMKL, Level Zero. No Vulkan builds, no new torch-XPU or IPEX code. Before
  proposing a build, check `ldd` for vulkan/torch/c10.
- **Engines are forks:** `engines/sd.cpp` (horvay/stable-diffusion.cpp `master`, with horvay/ggml `master`) and
  `engines/audio.cpp` (horvay/audio.cpp `main`). Commit there, push to those default branches (ggml first) only when
  Greg asks, then bump the submodule here.
- **Builds** run on the CPU with low parallelism (`JOBS`, default 3); this machine is shared. Watch RAM and disk.
- **Models** stay in `~/models` (`tools/mstore.ts`); never commit weights.
