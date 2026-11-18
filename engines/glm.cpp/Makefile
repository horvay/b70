# glm.cpp: run through ./build.sh (it sets up the DPC++ 2025.3 environment and the job count).
#
# Kernels are compiled ahead of time for the B70 (AOT=1, device bmg-g31): no driver JIT at start-up, and the ESIMD
# expert kernels ran faster AOT in the PyTorch stack. ESIMD needs the vector backend (-vc-codegen), which drops ordinary
# SYCL kernels' names, so the ESIMD files (EXL3 linear, routed experts) are their own library, libglm_esimd.so, and
# everything else is libglm.so. AOT=0: JIT (spir64) for both.
B ?= build
CXX := icpx
AOT ?= 1
DEV ?= bmg-g31
ifeq ($(AOT),1)
TGT := -fsycl-targets=spir64_gen
BE_ESIMD := -Xsycl-target-backend=spir64_gen "-device $(DEV) -options -vc-codegen"
BE_PF := -Xsycl-target-backend=spir64_gen "-device $(DEV) -options \"-vc-codegen -doubleGRF\""
BE_SYCL := -Xsycl-target-backend=spir64_gen "-device $(DEV)"
else
TGT := -fsycl-targets=spir64
endif
# fast math only for the ESIMD kernel files (as the PyTorch stack builds them); the rest masks with -inf, which
# -ffinite-math-only would compile away
MKLROOT ?= $(HOME)/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3
CXXFLAGS := -fsycl $(TGT) -fsycl-device-code-split=per_kernel -O3 -std=c++20 -Wall -fPIC -MMD -MP -DMKL_ILP64 \
            -I$(MKLROOT)/include
RPATH := -Wl,--disable-new-dtags,-rpath,$(CMPLR)/lib:$(UMF):$(MKLROOT)/lib:'$$ORIGIN'
LDFLAGS := -fsycl -static-intel $(RPATH) -L$(UMF) -lumf

ESIMD := $(B)/exl3.o $(B)/moe.o
# the 256-register library; Gemma's prompt attention masks with -inf: no fast math for it
PF := $(B)/moe_prefill.o $(B)/g4_fa.o $(B)/rp_fa.o $(B)/e0_dpas.o
CORE := $(B)/rt.o $(B)/progress.o $(B)/st.o $(B)/ops.o $(B)/kda.o $(B)/mla.o $(B)/moe_align.o $(B)/tier.o $(B)/lora.o $(B)/sampler.o $(B)/model.o $(B)/engine.o $(B)/http.o $(B)/q35.o $(B)/q35_ops.o $(B)/q35v.o $(B)/q4.o $(B)/q4_ops.o $(B)/gguf.o $(B)/e0.o $(B)/e0_ops.o $(B)/g4.o $(B)/g4_ops.o $(B)/g4_engine.o $(B)/rp.o $(B)/rp_ops.o
# host-only text code (tokenizer, chat template; PCRE2): no kernels, no libglm, linked into the programs that use it
HOST := $(B)/tokenizer.o $(B)/chat.o $(B)/chat_qwen.o $(B)/chat_gemma.o $(B)/chat_ripple.o $(B)/image.o
MKL := -L$(MKLROOT)/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core
LIBS := -L$(B) -lglm -lglm_esimd -lglm_moe_pf $(MKL)

all: $(B)/prio $(B)/exl3_check $(B)/decode_check $(B)/acts_check $(B)/moe_bench $(B)/tok_check $(B)/gemma_check $(B)/gen_bench \
     $(B)/glm_server $(B)/linear_bench $(B)/launch_bench $(B)/prefill_bench $(B)/q4_gen $(B)/e0_gen $(B)/e0_check $(B)/g4_gen $(B)/rp_gen

$(ESIMD) $(B)/moe_prefill.o: CXXFLAGS += -ffast-math
$(B)/tier.o: CXXFLAGS += -mavx2 -I$(HOME)/opt/onednn-src/third_party/level_zero
$(B)/rt.o: CXXFLAGS += -I$(HOME)/opt/onednn-src/third_party/level_zero
$(B)/sampler.o: CXXFLAGS += -mavx2 -mf16c
$(B)/image.o: CXXFLAGS += -I../sd.cpp/thirdparty -Wno-unused-function
$(B)/%.o: src/%.cpp | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(B)/t_%.o: tests/%.cpp | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(B)/libglm_moe_pf.so: $(PF)
	$(CXX) -fsycl $(TGT) $(BE_PF) -shared $^ -o $@ $(LDFLAGS)
$(B)/libglm_esimd.so: $(ESIMD) $(B)/libglm_moe_pf.so
	$(CXX) -fsycl $(TGT) $(BE_ESIMD) -shared $(ESIMD) -o $@ $(LDFLAGS) -L$(B) -lglm_moe_pf $(MKL)
$(B)/libglm.so: $(CORE) $(B)/libglm_esimd.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) -shared $(CORE) -o $@ $(LDFLAGS) -L$(B) -lglm_esimd -lglm_moe_pf $(MKL) -lze_loader

$(B)/prio: tools/prio.cpp | $(B)
	$(CXX) -fsycl $(TGT) $(BE_SYCL) -O3 -std=c++20 -Wall $< -o $@ $(LDFLAGS)
$(B)/launch_bench: tools/launch_bench.cpp | $(B)
	$(CXX) -fsycl $(TGT) $(BE_SYCL) -O3 -std=c++20 -Wall $< -o $@ $(LDFLAGS)
$(B)/x_%.o: tools/%.cpp | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(B)/glm_server: $(B)/x_glm_server.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_glm_server.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc -lze_loader
$(B)/q35_gen: $(B)/x_q35_gen.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_q35_gen.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/q35v_check: $(B)/t_q35v_check.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/t_q35v_check.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/g4_gen: $(B)/x_g4_gen.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_g4_gen.o -o $@ $(LDFLAGS) $(LIBS) -lze_loader
# llm/g4draft: the drafter trainer (its kernels in g4_draft.o, not in libglm)
$(B)/g4_draft: $(B)/x_g4_draft.o $(B)/g4_draft.o $(B)/tokenizer.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_g4_draft.o $(B)/g4_draft.o $(B)/tokenizer.o -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc -lze_loader
$(B)/rp_gen: $(B)/x_rp_gen.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_rp_gen.o -o $@ $(LDFLAGS) $(LIBS) -lze_loader
$(B)/q4_gen: $(B)/x_q4_gen.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_q4_gen.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/e0_gen: $(B)/x_e0_gen.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_e0_gen.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/e0_check: $(B)/t_e0_check.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/t_e0_check.o -o $@ $(LDFLAGS) $(LIBS)
$(B)/linear_bench: $(B)/x_linear_bench.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_linear_bench.o -o $@ $(LDFLAGS) $(LIBS)
$(B)/gen_bench: $(B)/x_gen_bench.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_gen_bench.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/prefill_bench: $(B)/x_prefill_bench.o $(HOST) $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $(B)/x_prefill_bench.o $(HOST) -o $@ $(LDFLAGS) $(LIBS) -lpcre2-8 -lutf8proc
$(B)/mkl_lora_repro: tools/mkl_lora_repro.cpp | $(B)
	$(CXX) $(CXXFLAGS) $(BE_SYCL) $< -o $@ $(LDFLAGS) $(MKL)
ZEINC := $(HOME)/opt/onednn-src/third_party/level_zero
$(B)/x_udmabuf_test.o: CXXFLAGS += -I$(ZEINC)
$(B)/udmabuf_test: $(B)/x_udmabuf_test.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $< -o $@ $(LDFLAGS) $(LIBS) -lze_loader
$(B)/moe_bench: $(B)/t_moe_bench.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $< -o $@ $(LDFLAGS) $(LIBS)
$(B)/tok_check: $(B)/t_tok_check.o $(HOST)
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $^ -o $@ $(LDFLAGS) -lpcre2-8 -lutf8proc
$(B)/q4_chat_check: $(B)/t_q4_chat_check.o $(filter-out $(B)/image.o,$(HOST))
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $^ -o $@ $(LDFLAGS) -lpcre2-8 -lutf8proc
$(B)/gemma_check: $(B)/t_gemma_check.o $(filter-out $(B)/image.o,$(HOST))
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $^ -o $@ $(LDFLAGS) -lpcre2-8 -lutf8proc
$(B)/ripple_check: $(B)/t_ripple_check.o $(filter-out $(B)/image.o,$(HOST))
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $^ -o $@ $(LDFLAGS) -lpcre2-8 -lutf8proc
$(B)/%_check: $(B)/t_%_check.o $(B)/libglm.so
	$(CXX) -fsycl $(TGT) $(BE_SYCL) $< -o $@ $(LDFLAGS) $(LIBS)

$(B):
	mkdir -p $(B)

-include $(wildcard $(B)/*.d)
.PHONY: all
