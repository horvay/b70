#include "dit-fusion.hpp"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <vector>

// A row of a broadcast operand (ggml_can_repeat onto the full shape), indexed by the full tensor's row coordinates.
// Only the row dims broadcast: every operand here spans the full ne0 contiguously.
struct dit_bcast {
    const char * data;
    int64_t      ne1, ne2, ne3;
    size_t       nb1, nb2, nb3;

    const float * row(int64_t i1, int64_t i2, int64_t i3) const {
        return (const float *) (data + (ne1 == 1 ? 0 : i1 % ne1) * nb1 + (ne2 == 1 ? 0 : i2 % ne2) * nb2 +
                                (ne3 == 1 ? 0 : i3 % ne3) * nb3);
    }
};

static dit_bcast dit_bcast_of(const ggml_tensor * t) {
    return { (const char *) t->data, t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3] };
}

// f32, full rows (ne0 of the fused shape, unit stride), repeatable onto `full`
static bool dit_row_operand_ok(const ggml_tensor * t, const ggml_tensor * full) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == full->ne[0] && t->nb[0] == sizeof(float) && ggml_can_repeat(t, full);
}

// the operand of binary node `n` that is not `chain`: the broadcast src1, or src0 when it has the full shape (ggml
// broadcasts src1 only, so a chain value in src1 means src0 is full size); nullptr if `chain` is not a src of `n`
static const ggml_tensor * dit_other(const ggml_tensor * n, const ggml_tensor * chain) {
    if (n->src[0] == chain) {
        return n->src[1];
    }
    if (n->src[1] == chain && ggml_are_same_shape(n->src[0], n)) {
        return n->src[0];
    }
    return nullptr;
}

static bool dit_is_noop(const ggml_tensor * n) {
    return ggml_is_empty(n) || n->op == GGML_OP_NONE || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
}

static int dit_index_of(const ggml_cgraph * cgraph, int lo, int hi, const ggml_tensor * t) {
    for (int j = lo; j <= hi && j < cgraph->n_nodes; j++) {
        if (cgraph->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

// The fused kernel writes `root` and replaces nodes [first, root]: every compute node in that span must be part of the
// pattern (`inner` + root; views and no-ops elsewhere in the span compute nothing and are skipped harmlessly), `first`
// must be the pattern's earliest node, and nothing outside the pattern may read an intermediate. Returns root's index
// or -1.
static int dit_check_span(const ggml_cgraph * cgraph, int first, const ggml_tensor * root,
                          const std::vector<const ggml_tensor *> & inner) {
    constexpr int window = 16;
    const int     last   = dit_index_of(cgraph, first, first + window, root);
    if (last < 0 || (root->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return -1;
    }
    auto in_pattern = [&](const ggml_tensor * t) {
        if (t == root) {
            return true;
        }
        for (const ggml_tensor * p : inner) {
            if (p == t) {
                return true;
            }
        }
        return false;
    };
    bool first_in = false;
    for (int j = first; j <= last; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (in_pattern(n)) {
            first_in |= j == first;
            continue;
        }
        if (!dit_is_noop(n) && (n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return -1;
        }
    }
    if (!first_in) {
        return -1;
    }
    for (const ggml_tensor * p : inner) {
        const int j = dit_index_of(cgraph, first, last, p);
        if (j < 0 || (p->flags & GGML_TENSOR_FLAG_OUTPUT) || (p->op != GGML_OP_VIEW && (p->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
            return -1;
        }
        int uses = 0;
        for (const ggml_tensor * q : inner) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                uses += q->src[s] == p;
            }
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uses += root->src[s] == p;
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) {
            return -1;
        }
    }
    return last;
}

static float dit_op_param_f32(const ggml_tensor * t, int i) {
    float v;
    memcpy(&v, (const float *) t->op_params + i, sizeof(float));
    return v;
}

// The allocator may give the fused output memory that an input still occupies: it treats an input as dead once the
// last pattern node reading it would have run, and those nodes never run. Only element-for-element aliasing (same data,
// shape and strides) is safe, since every kernel here reads an element before the same work-item writes it.
static bool dit_alias_ok(const ggml_tensor * out, std::initializer_list<const ggml_tensor *> ins) {
    const char * o0 = (const char *) out->data;
    const char * o1 = o0 + ggml_nbytes(out);
    for (const ggml_tensor * in : ins) {
        if (in == nullptr) {
            continue;
        }
        const char * i0 = (const char *) in->data;
        const char * i1 = i0 + ggml_nbytes(in);
        if (o1 <= i0 || i1 <= o0) {
            continue;
        }
        if (in->data != out->data || !ggml_are_same_shape(in, out) || in->nb[1] != out->nb[1] || in->nb[2] != out->nb[2] ||
            in->nb[3] != out->nb[3]) {
            return false;
        }
    }
    return true;
}

// a LoRA output added to its base matmul (sd.cpp's add_out_diff: ggml_add_inplace(out, diff)): both operands full
// shape, f32, contiguous, so the diff can be read at the same element offsets as the sum
static bool dit_is_lora_add(const ggml_tensor * a) {
    return a->op == GGML_OP_ADD && a->type == GGML_TYPE_F32 && a->src[0]->type == GGML_TYPE_F32 &&
           a->src[1]->type == GGML_TYPE_F32 && ggml_are_same_shape(a->src[0], a) && ggml_are_same_shape(a->src[1], a) &&
           ggml_is_contiguous(a) && ggml_is_contiguous(a->src[0]) && ggml_is_contiguous(a->src[1]);
}

// ---------------------------------------------------------------------------------------------------------------------
// norm_mod: dst = rms_norm(x) * w * m1 + m0, one work-group per row

// sd.cpp's RMSNorm multiplies by its weight in place (the MUL is a view of the norm), which ggml_can_fuse rejects, so
// the chain is checked with dit_check_span like the others. Returns the ADD's index or -1.
static int dit_match_norm_mod(const ggml_cgraph * cgraph, int i, const ggml_tensor ** mw, const ggml_tensor ** m1) {
    const ggml_tensor * rms = cgraph->nodes[i];
    const ggml_tensor * x   = rms->src[0];
    if (rms->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || i + 3 >= cgraph->n_nodes) {
        return -1;
    }
    // the three ops follow each other directly (no views in between)
    const ggml_tensor * chain[3] = { cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3] };
    const ggml_op       ops[3]   = { GGML_OP_MUL, GGML_OP_MUL, GGML_OP_ADD };
    const ggml_tensor * prev     = rms;
    for (int k = 0; k < 3; k++) {
        const ggml_tensor * n = chain[k];
        const ggml_tensor * o = n->op == ops[k] ? dit_other(n, prev) : nullptr;
        if (n->type != GGML_TYPE_F32 || o == nullptr || !ggml_are_same_shape(n, rms) || !dit_row_operand_ok(o, n)) {
            return -1;
        }
        prev = n;
    }
    if (!ggml_is_contiguous(chain[2])) {
        return -1;
    }
    if (!dit_alias_ok(chain[2], { x, dit_other(chain[0], rms), dit_other(chain[1], chain[0]), dit_other(chain[2], chain[1]) })) {
        return -1;
    }
    *mw = chain[0];
    *m1 = chain[1];
    return dit_check_span(cgraph, i, chain[2], { rms, chain[0], chain[1] });
}

static void dit_norm_mod(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_tensor * m_w,
                         const ggml_tensor * m_1, ggml_tensor * dst) {
    const ggml_tensor * rms = cgraph->nodes[i];
    const ggml_tensor * x   = rms->src[0];
    const dit_bcast     w   = dit_bcast_of(dit_other(m_w, rms));
    const dit_bcast     b1  = dit_bcast_of(dit_other(m_1, m_w));
    const dit_bcast     b0  = dit_bcast_of(dit_other(dst, m_1));
    const float         eps = dit_op_param_f32(rms, 0);

    const int64_t C = x->ne[0], R1 = x->ne[1], R2 = x->ne[2], R3 = x->ne[3];
    const size_t  nb1 = x->nb[1], nb2 = x->nb[2], nb3 = x->nb[3];
    const char *  xd  = (const char *) x->data;
    float *       dd  = (float *) dst->data;

    constexpr int WG  = 256;
    constexpr int NSG = WG / WARP_SIZE;
    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> s_sum(sycl::range<1>(NSG), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(R3, R2, R1 * WG), sycl::range<3>(1, 1, WG)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             const int64_t i1 = it.get_group(2), i2 = it.get_group(1), i3 = it.get_group(0);
                             const int     tid = it.get_local_id(2);
                             const float * xr  = (const float *) (xd + i1 * nb1 + i2 * nb2 + i3 * nb3);
                             float         ss  = 0.f;
                             for (int64_t c = tid; c < C; c += WG) {
                                 const float v = xr[c];
                                 ss += v * v;
                             }
                             ss = warp_reduce_sum(ss, it);
                             const auto sg = it.get_sub_group();
                             if (sg.get_local_linear_id() == 0) {
                                 s_sum[sg.get_group_linear_id()] = ss;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             ss = 0.f;
#pragma unroll
                             for (int k = 0; k < NSG; k++) {
                                 ss += s_sum[k];
                             }
                             const float   scale = sycl::rsqrt(ss / C + eps);
                             const float * wr    = w.row(i1, i2, i3);
                             const float * r1    = b1.row(i1, i2, i3);
                             const float * r0    = b0.row(i1, i2, i3);
                             float *       dr    = dd + ((i3 * R2 + i2) * R1 + i1) * C;
                             for (int64_t c = tid; c < C; c += WG) {
                                 dr[c] = scale * xr[c] * wr[c] * r1[c] + r0[c];
                             }
                         });
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// norm_rope: dst = n * rc + swap_halves(n) * rs, n = rms_norm(x) * w, over rows of D (one head of one token). One
// sub-group per row, D / WARP_SIZE values per lane: lane l holds elements l + k * WARP_SIZE, so element j's swap
// partner j +- D/2 sits in the same lane, NV/2 slots away.

struct dit_rope_match {
    int                 last;
    const ggml_tensor * rms;
    const ggml_tensor * w;
    const ggml_tensor * rc;
    const ggml_tensor * rs;
};

static bool dit_match_norm_rope(const ggml_cgraph * cgraph, int i, dit_rope_match & m) {
    const ggml_tensor * rms = cgraph->nodes[i];
    if (rms->op != GGML_OP_RMS_NORM || rms->type != GGML_TYPE_F32 || rms->src[0]->type != GGML_TYPE_F32 ||
        rms->src[0]->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t D = rms->ne[0];
    if (D != 64 && D != 128 && D != 256) {
        return false;
    }
    const ggml_tensor *mw = nullptr, *lo = nullptr, *hi = nullptr, *cat = nullptr, *a = nullptr, *b = nullptr, *root = nullptr;
    const int          end = std::min(cgraph->n_nodes - 1, i + 16);
    for (int j = i + 1; j <= end; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!mw) {
            if (n->op == GGML_OP_MUL && dit_other(n, rms)) {
                mw = n;
            }
            continue;
        }
        if (n->op == GGML_OP_VIEW && n->src[0] == mw && n->ne[0] == D / 2 && n->ne[1] == mw->ne[1] && n->ne[2] == mw->ne[2] &&
            n->ne[3] == mw->ne[3] && n->nb[1] == mw->nb[1] && n->nb[2] == mw->nb[2] && n->nb[3] == mw->nb[3]) {
            if (n->view_offs == 0) {
                lo = n;
            } else if (n->view_offs == (size_t) (D / 2) * sizeof(float)) {
                hi = n;
            }
        } else if (n->op == GGML_OP_CONCAT && hi && lo && n->src[0] == hi && n->src[1] == lo &&
                   ggml_get_op_params_i32(n, 0) == 0) {
            cat = n;
        } else if (n->op == GGML_OP_MUL && n->src[0] == mw) {
            a = n;
        } else if (n->op == GGML_OP_MUL && cat && n->src[0] == cat) {
            b = n;
        } else if (n->op == GGML_OP_ADD && a && b &&
                   ((n->src[0] == a && n->src[1] == b) || (n->src[0] == b && n->src[1] == a))) {
            root = n;
            break;
        }
    }
    if (!root || !ggml_is_contiguous(root) || !ggml_is_contiguous(mw)) {
        return false;
    }
    const ggml_tensor * w = dit_other(mw, rms);
    if (!dit_row_operand_ok(w, mw) || !dit_row_operand_ok(a->src[1], a) || !dit_row_operand_ok(b->src[1], b) ||
        mw->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || root->type != GGML_TYPE_F32) {
        return false;
    }
    if (!dit_alias_ok(root, { rms->src[0], w, a->src[1], b->src[1] })) {
        return false;
    }
    m.last = dit_check_span(cgraph, i, root, { rms, mw, lo, hi, cat, a, b });
    m.rms  = rms;
    m.w    = w;
    m.rc   = a->src[1];
    m.rs   = b->src[1];
    return m.last >= 0;
}

template <int D>
static void dit_norm_rope_launch(ggml_backend_sycl_context & ctx, const dit_rope_match & m, float * dd) {
    constexpr int NV   = D / WARP_SIZE;
    constexpr int WG   = 256;
    constexpr int ROWS = WG / WARP_SIZE;
    const ggml_tensor * x = m.rms->src[0];
    const int64_t H = x->ne[1], L = x->ne[2], N = x->ne[3], R = H * L * N;
    const size_t  nb1 = x->nb[1], nb2 = x->nb[2], nb3 = x->nb[3];
    const char *  xd  = (const char *) x->data;
    const float   eps = dit_op_param_f32(m.rms, 0);
    const dit_bcast w = dit_bcast_of(m.w), rc = dit_bcast_of(m.rc), rs = dit_bcast_of(m.rs);
    const int64_t ngroups = (R + ROWS - 1) / ROWS;
    ctx.stream()->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * WG), sycl::range<3>(1, 1, WG)),
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const auto    sg = it.get_sub_group();
            const int64_t r  = it.get_group(2) * ROWS + sg.get_group_linear_id();
            if (r >= R) {
                return;
            }
            const int     lane = sg.get_local_linear_id();
            const int64_t i1 = r % H, i2 = (r / H) % L, i3 = r / (H * L);
            const float * xr = (const float *) (xd + i1 * nb1 + i2 * nb2 + i3 * nb3);
            float         v[NV];
            float         ss = 0.f;
#pragma unroll
            for (int k = 0; k < NV; k++) {
                v[k] = xr[lane + k * WARP_SIZE];
                ss += v[k] * v[k];
            }
            ss                = warp_reduce_sum(ss, it);
            const float scale = sycl::rsqrt(ss / D + eps);
            const float * wr  = w.row(i1, i2, i3);
#pragma unroll
            for (int k = 0; k < NV; k++) {
                v[k] = scale * v[k] * wr[lane + k * WARP_SIZE];
            }
            const float * cr = rc.row(i1, i2, i3);
            const float * sr = rs.row(i1, i2, i3);
            float *       dr = dd + r * D;
#pragma unroll
            for (int k = 0; k < NV; k++) {
                const int j = lane + k * WARP_SIZE;
                dr[j]       = v[k] * cr[j] + v[(k + NV / 2) % NV] * sr[j];
            }
        });
}

static void dit_norm_rope(ggml_backend_sycl_context & ctx, const dit_rope_match & m, ggml_tensor * root) {
    float * dd = (float *) root->data;
    switch (m.rms->ne[0]) {
        case 64:  dit_norm_rope_launch<64>(ctx, m, dd); break;
        case 128: dit_norm_rope_launch<128>(ctx, m, dd); break;
        case 256: dit_norm_rope_launch<256>(ctx, m, dd); break;
        default:  GGML_ABORT("unreachable");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// situ: dst = tanh(sa * g + ba) * sigmoid(g) * tanh(sb * h + bb), g and h strided (the halves of one fused matmul). With a
// LoRA on that matmul, g and h are views of the LoRA add (base + diff), which is folded in: g = gb + gd, h = hb + hd.

struct dit_situ_match {
    int                 last;
    const ggml_tensor * g;
    const ggml_tensor * h;
    const ggml_tensor * diff;  // the LoRA diff, or nullptr
    const ggml_tensor * sum;   // the LoRA add g and h view, or nullptr
    float               sa, ba, sb, bb;
};

static bool dit_is_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// mul(tanh(scale(g)), sigmoid(g)), in either operand order; returns g or nullptr
static const ggml_tensor * dit_situ_gate(const ggml_tensor * a, const ggml_tensor ** t, const ggml_tensor ** s,
                                         const ggml_tensor ** sg) {
    if (!a || a->op != GGML_OP_MUL) {
        return nullptr;
    }
    for (int o = 0; o < 2; o++) {
        const ggml_tensor * tt = a->src[o];
        const ggml_tensor * ss = a->src[1 - o];
        if (dit_is_unary(tt, GGML_UNARY_OP_TANH) && tt->src[0]->op == GGML_OP_SCALE && dit_is_unary(ss, GGML_UNARY_OP_SIGMOID) &&
            tt->src[0]->src[0] == ss->src[0]) {
            *t  = tt;
            *s  = tt->src[0];
            *sg = ss;
            return ss->src[0];
        }
    }
    return nullptr;
}

// i is either the pattern's first op (SCALE or SIGMOID) or a LoRA add whose two views feed it
static bool dit_match_situ(const ggml_cgraph * cgraph, int i, dit_situ_match & m) {
    const ggml_tensor * la    = nullptr;
    int                 first = i;
    if (cgraph->nodes[i]->op == GGML_OP_ADD) {
        la = cgraph->nodes[i];
        if (!dit_is_lora_add(la)) {
            return false;
        }
        first = i + 1;
        while (first < cgraph->n_nodes && dit_is_noop(cgraph->nodes[first])) {
            first++;
        }
        if (first >= cgraph->n_nodes) {
            return false;
        }
    }
    const ggml_tensor * f = cgraph->nodes[first];
    if (f->op != GGML_OP_SCALE && !dit_is_unary(f, GGML_UNARY_OP_SIGMOID)) {
        return false;
    }
    const int end = std::min(cgraph->n_nodes - 1, first + 16);
    for (int j = first + 2; j <= end; j++) {
        const ggml_tensor * root = cgraph->nodes[j];
        if (root->op != GGML_OP_MUL) {
            continue;
        }
        for (int o = 0; o < 2; o++) {
            const ggml_tensor *t1 = nullptr, *s1 = nullptr, *sg = nullptr;
            const ggml_tensor * a  = root->src[o];
            const ggml_tensor * t2 = root->src[1 - o];
            const ggml_tensor * g  = dit_situ_gate(a, &t1, &s1, &sg);
            if (!g || !dit_is_unary(t2, GGML_UNARY_OP_TANH) || t2->src[0]->op != GGML_OP_SCALE) {
                continue;
            }
            const ggml_tensor * s2 = t2->src[0];
            const ggml_tensor * h  = s2->src[0];
            // all one shape, f32, unit-stride inputs, contiguous output
            const ggml_tensor * all[] = { g, h, s1, t1, sg, a, s2, t2, root };
            bool                ok    = ggml_is_contiguous(root) && g->nb[0] == sizeof(float) && h->nb[0] == sizeof(float);
            for (const ggml_tensor * n : all) {
                ok = ok && n->type == GGML_TYPE_F32 && ggml_are_same_shape(n, root);
            }
            std::vector<const ggml_tensor *> inner = { s1, t1, sg, a, s2, t2 };
            if (la) {
                // g and h must be views of the LoRA add (which then only feeds them)
                ok = ok && g->op == GGML_OP_VIEW && h->op == GGML_OP_VIEW && g->src[0] == la && h->src[0] == la && g != h;
                inner.insert(inner.end(), { la, g, h });
            }
            if (!ok || !dit_alias_ok(root, { g, h, la ? la->src[0] : nullptr, la ? la->src[1] : nullptr })) {
                continue;
            }
            m.last = dit_check_span(cgraph, i, root, inner);
            if (m.last < 0) {
                continue;
            }
            m.g    = g;
            m.h    = h;
            m.sum  = la;
            m.diff = la ? la->src[1] : nullptr;
            m.sa   = dit_op_param_f32(s1, 0);
            m.ba   = dit_op_param_f32(s1, 1);
            m.sb   = dit_op_param_f32(s2, 0);
            m.bb   = dit_op_param_f32(s2, 1);
            return true;
        }
    }
    return false;
}

template <bool lora>
static void dit_situ_launch(ggml_backend_sycl_context & ctx, const dit_situ_match & m, ggml_tensor * root) {
    const int64_t ne0 = root->ne[0], ne1 = root->ne[1], ne2 = root->ne[2], ne3 = root->ne[3];
    // with a LoRA, g and h point into the add's (in place: the base matmul's) data; the diff has the same layout, so
    // its elements sit at the same byte offsets from the diff's start
    const char * gd = (const char *) m.g->data;
    const char * hd = (const char *) m.h->data;
    const char * gx = lora ? (const char *) m.diff->data + ((const char *) m.g->data - (const char *) m.sum->data) : nullptr;
    const char * hx = lora ? (const char *) m.diff->data + ((const char *) m.h->data - (const char *) m.sum->data) : nullptr;
    if (lora) {
        // the base operand of the add holds the base values (the add itself never runs)
        gd = (const char *) m.sum->src[0]->data + ((const char *) m.g->data - (const char *) m.sum->data);
        hd = (const char *) m.sum->src[0]->data + ((const char *) m.h->data - (const char *) m.sum->data);
    }
    const size_t gnb1 = m.g->nb[1], gnb2 = m.g->nb[2], gnb3 = m.g->nb[3];
    const size_t hnb1 = m.h->nb[1], hnb2 = m.h->nb[2], hnb3 = m.h->nb[3];
    const float  sa = m.sa, ba = m.ba, sb = m.sb, bb = m.bb;
    float *      dd = (float *) root->data;
    constexpr int64_t WG      = 256;
    const int64_t     ne0_pad = (ne0 + WG - 1) / WG * WG;
    ctx.stream()->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(ne2 * ne3, ne1, ne0_pad), sycl::range<3>(1, 1, WG)), [=](sycl::nd_item<3> it) {
            const int64_t i0 = it.get_global_id(2);
            if (i0 >= ne0) {
                return;
            }
            const int64_t i1 = it.get_global_id(1), i23 = it.get_global_id(0);
            const int64_t i2 = ne3 == 1 ? i23 : i23 % ne2;
            const int64_t i3 = ne3 == 1 ? 0 : i23 / ne2;
            const size_t  go = i1 * gnb1 + i2 * gnb2 + i3 * gnb3;
            const size_t  ho = i1 * hnb1 + i2 * hnb2 + i3 * hnb3;
            float         g  = ((const float *) (gd + go))[i0];
            float         h  = ((const float *) (hd + ho))[i0];
            if constexpr (lora) {
                g += ((const float *) (gx + go))[i0];
                h += ((const float *) (hx + ho))[i0];
            }
            const float sig = 1.f / (1.f + sycl::exp(-g));
            dd[((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0] = sycl::tanh(sa * g + ba) * sig * sycl::tanh(sb * h + bb);
        });
}

static void dit_situ(ggml_backend_sycl_context & ctx, const dit_situ_match & m, ggml_tensor * root) {
    if (m.sum) {
        dit_situ_launch<true>(ctx, m, root);
    } else {
        dit_situ_launch<false>(ctx, m, root);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// mul_add: dst = x + y * g, g broadcast over rows (a gated residual). With a LoRA on the matmul producing y, y is the
// LoRA add (base + diff), which is folded in: dst = x + (base + diff) * g.

struct dit_mul_add_match {
    int                 last;
    const ggml_tensor * y;     // y, or the base operand of the LoRA add
    const ggml_tensor * diff;  // the LoRA diff, or nullptr
    const ggml_tensor * x;
    const ggml_tensor * g;
};

// i is the MUL, or a LoRA add followed by the MUL
static bool dit_match_mul_add(const ggml_cgraph * cgraph, int i, dit_mul_add_match & m) {
    const ggml_tensor * la = nullptr;
    int                 k  = i;
    if (cgraph->nodes[i]->op == GGML_OP_ADD) {
        la = cgraph->nodes[i];
        k  = i + 1;
        if (!dit_is_lora_add(la) || k >= cgraph->n_nodes || cgraph->nodes[k]->src[0] != la) {
            return false;
        }
    }
    if (!ggml_can_fuse(cgraph, k, { GGML_OP_MUL, GGML_OP_ADD })) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[k];
    const ggml_tensor * add = cgraph->nodes[k + 1];
    const ggml_tensor * y   = mul->src[0];
    const ggml_tensor * g   = mul->src[1];
    const ggml_tensor * x   = add->src[0] == mul ? add->src[1] : add->src[0];
    const bool ok = mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && y->nb[0] == sizeof(float) &&
                    ggml_are_same_shape(y, mul) && ggml_are_same_shape(x, add) && x->type == GGML_TYPE_F32 &&
                    x->nb[0] == sizeof(float) && dit_row_operand_ok(g, mul) && ggml_is_contiguous(add);
    if (!ok) {
        return false;
    }
    if (la) {
        // the add only feeds the MUL
        if (dit_check_span(cgraph, i, add, { la, mul }) < 0) {
            return false;
        }
        y = la->src[0];
    }
    if (!dit_alias_ok(add, { y, x, g, la ? la->src[1] : nullptr })) {
        return false;
    }
    m.last = k + 1;
    m.y    = y;
    m.diff = la ? la->src[1] : nullptr;
    m.x    = x;
    m.g    = g;
    return true;
}

template <bool lora>
static void dit_mul_add_launch(ggml_backend_sycl_context & ctx, const dit_mul_add_match & m, ggml_tensor * add) {
    const dit_bcast g   = dit_bcast_of(m.g);
    const int64_t   ne0 = add->ne[0], ne1 = add->ne[1], ne2 = add->ne[2], ne3 = add->ne[3];
    const char *    yd  = (const char *) m.y->data;
    const char *    zd  = lora ? (const char *) m.diff->data : nullptr;  // same layout as y (both contiguous)
    const char *    xd  = (const char *) m.x->data;
    const size_t    ynb1 = m.y->nb[1], ynb2 = m.y->nb[2], ynb3 = m.y->nb[3];
    const size_t    xnb1 = m.x->nb[1], xnb2 = m.x->nb[2], xnb3 = m.x->nb[3];
    float *         dd   = (float *) add->data;
    constexpr int64_t WG      = 256;
    const int64_t     ne0_pad = (ne0 + WG - 1) / WG * WG;
    ctx.stream()->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(ne2 * ne3, ne1, ne0_pad), sycl::range<3>(1, 1, WG)), [=](sycl::nd_item<3> it) {
            const int64_t i0 = it.get_global_id(2);
            if (i0 >= ne0) {
                return;
            }
            const int64_t i1 = it.get_global_id(1), i23 = it.get_global_id(0);
            const int64_t i2 = ne3 == 1 ? i23 : i23 % ne2;
            const int64_t i3 = ne3 == 1 ? 0 : i23 / ne2;
            const size_t  yo = i1 * ynb1 + i2 * ynb2 + i3 * ynb3;
            float         yv = ((const float *) (yd + yo))[i0];
            if constexpr (lora) {
                yv += ((const float *) (zd + yo))[i0];
            }
            const float xv = ((const float *) (xd + i1 * xnb1 + i2 * xnb2 + i3 * xnb3))[i0];
            dd[((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0] = xv + yv * g.row(i1, i2, i3)[i0];
        });
}

static void dit_mul_add(ggml_backend_sycl_context & ctx, const dit_mul_add_match & m, ggml_tensor * add) {
    if (m.diff) {
        dit_mul_add_launch<true>(ctx, m, add);
    } else {
        dit_mul_add_launch<false>(ctx, m, add);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// lora_gemm_acc: a LoRA up-projection u = up @ lx whose only use is an in-place add into the base matmul's output (ADD,
// or ACC into one slice of a fused projection, from sd.cpp's add_out_diff) -> one GEMM accumulating into that output
// (beta = 1). The diff is never written, and the full-width add disappears. Anchored at the GEMM, so every tensor it
// touches is still live in the allocator's plan.

struct dit_gemm_acc_match {
    const ggml_tensor * a;  // up weights [K, M]
    const ggml_tensor * b;  // lx [K, N]
    float *             c;  // where u's [M, N] lands in the base output
    int64_t             ldc;
};

static bool dit_match_gemm_acc(const ggml_cgraph * cgraph, int i, dit_gemm_acc_match & m) {
    const ggml_tensor * u = cgraph->nodes[i];
    if (u->op != GGML_OP_MUL_MAT || i + 1 >= cgraph->n_nodes || (u->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_node_get_use_count(cgraph, i) != 1) {
        return false;
    }
    const ggml_tensor * a = u->src[0];
    const ggml_tensor * b = u->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) ||
        b->nb[0] != sizeof(float) || a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * n = cgraph->nodes[i + 1];
    if (n->src[1] != u || n->type != GGML_TYPE_F32 || n->data != n->src[0]->data || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;  // not an in-place accumulation of u
    }
    if (n->op == GGML_OP_ADD) {
        if (!ggml_are_same_shape(n, u) || n->nb[0] != sizeof(float) || n->ne[2] != 1 || n->ne[3] != 1) {
            return false;
        }
        m.c   = (float *) n->data;
        m.ldc = n->nb[1] / sizeof(float);
    } else if (n->op == GGML_OP_ACC) {
        const int32_t * p = (const int32_t *) n->op_params;  // nb1, nb2, nb3, offset (bytes), inplace
        if (n->nb[0] != sizeof(float) || p[0] % sizeof(float) != 0 || p[3] % sizeof(float) != 0 || u->ne[0] * sizeof(float) > (size_t) p[0]) {
            return false;
        }
        m.c   = (float *) ((char *) n->data + p[3]);
        m.ldc = p[0] / sizeof(float);
    } else {
        return false;
    }
    m.a = a;
    m.b = b;
    return true;
}

// LoRA weights load as f32, so their matmuls run on the f32 path at a fraction of what the matrix engines do in bf16 or
// tf32; a LoRA diff does not need f32 inputs. GGML_SYCL_LORA_GEMM_MODE: oneMKL compute_mode bits, default float_to_bf16
// (0x1; on the B70 with the MOSS turbo LoRA: 8 steps 14.85 s -> 12.56 s, image unchanged). 0 = plain f32 (the down
// projections then stay on the regular MUL_MAT path); tf32 (0x10) is no faster there, and forcing it throws.
static oneapi::mkl::blas::compute_mode dit_lora_gemm_mode() {
    static const auto mode = [] {
        const char * e = getenv("GGML_SYCL_LORA_GEMM_MODE");
        return (oneapi::mkl::blas::compute_mode) (e ? strtoull(e, nullptr, 0) : 0x1);
    }();
    return mode;
}

// ggml mul_mat: c[M, N] (+)= a[K, M]^T b[K, N]; column major with c rows ldc apart
static void dit_lora_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * a, const ggml_tensor * b, float * c, int64_t ldc,
                          float beta) {
    const int64_t K = a->ne[0], M = a->ne[1], N = b->ne[1];
    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *ctx.stream(), oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, M, N, K, 1.0f, (const float *) a->data,
        a->nb[1] / sizeof(float), (const float *) b->data, b->nb[1] / sizeof(float), beta, c, ldc, dit_lora_gemm_mode())));
}

static void dit_gemm_acc(ggml_backend_sycl_context & ctx, const dit_gemm_acc_match & m) {
    dit_lora_gemm(ctx, m.a, m.b, m.c, m.ldc, 1.0f);
}

// any other f32 LoRA matmul (the rank-r down projections): the same GEMM, plain store
static bool dit_match_lora_gemm(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * u = cgraph->nodes[i];
    const ggml_tensor * a = u->src[0];
    const ggml_tensor * b = u->src[1];
    return dit_lora_gemm_mode() != (oneapi::mkl::blas::compute_mode) 0 && strstr(a->name, ".lora_") != nullptr &&
           a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && u->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
           ggml_is_contiguous(u) && b->nb[0] == sizeof(float) && a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 &&
           b->ne[3] == 1;
}

// ---------------------------------------------------------------------------------------------------------------------

int ggml_sycl_try_dit_fusion(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i, const char ** label) {
    static const bool enabled = g_ggml_sycl_enable_fusion && getenv("GGML_SYCL_NO_DIT_FUSION") == nullptr;
    if (!enabled) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op == GGML_OP_MUL_MAT) {
        static const bool gemm_acc = getenv("GGML_SYCL_NO_LORA_GEMM_ACC") == nullptr;
        dit_gemm_acc_match gm;
        if (gemm_acc && dit_match_gemm_acc(cgraph, i, gm)) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            dit_gemm_acc(ctx, gm);
            *label = "FUSED lora_gemm_acc";
            return 1;
        }
        if (dit_match_lora_gemm(cgraph, i)) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            ggml_tensor * u = cgraph->nodes[i];
            dit_lora_gemm(ctx, u->src[0], u->src[1], (float *) u->data, u->ne[0], 0.0f);
            *label = "FUSED lora_gemm";
            // a node of its own: report it as consumed without skipping anything after it
            return -1;
        }
        return 0;
    }
    if (node->op == GGML_OP_RMS_NORM) {
        dit_rope_match rm;
        if (dit_match_norm_rope(cgraph, i, rm)) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            dit_norm_rope(ctx, rm, cgraph->nodes[rm.last]);
            *label = "FUSED norm_rope";
            return rm.last - i;
        }
        const ggml_tensor *mw = nullptr, *m1 = nullptr;
        const int          last = dit_match_norm_mod(cgraph, i, &mw, &m1);
        if (last >= 0) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            dit_norm_mod(ctx, cgraph, i, mw, m1, cgraph->nodes[last]);
            *label = "FUSED norm_mod";
            return last - i;
        }
        return 0;
    }
    if (node->op == GGML_OP_SCALE || node->op == GGML_OP_UNARY || node->op == GGML_OP_ADD) {
        dit_situ_match sm;
        if (dit_match_situ(cgraph, i, sm)) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            dit_situ(ctx, sm, cgraph->nodes[sm.last]);
            *label = sm.sum ? "FUSED lora_situ" : "FUSED situ";
            return sm.last - i;
        }
    }
    if (node->op == GGML_OP_MUL || node->op == GGML_OP_ADD) {
        dit_mul_add_match mm;
        if (dit_match_mul_add(cgraph, i, mm)) {
            SYCL_CHECK(ggml_sycl_set_device(ctx.device));
            dit_mul_add(ctx, mm, cgraph->nodes[mm.last]);
            *label = mm.diff ? "FUSED lora_mul_add" : "FUSED mul_add";
            return mm.last - i;
        }
    }
    return 0;
}
