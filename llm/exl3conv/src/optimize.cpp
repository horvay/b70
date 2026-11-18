// exl3opt: per-group bit allocation from exl3measure's sensitivities (exllamav3's sc_optimize.py,
// notes/optimize-recipes.md §4), CPU only.
//
//   exl3opt -m noise.json -a rfn.json -b 4.0 -o recipe.json [--min-k 3] [--max-k 6] [--alpha 2.0]
//
// Model: a target's KL grows as S_t * rfn_t(K)^alpha, rfn_t(K) = anchor rfn * 1.96^(anchor K - K) (-a: exl3eval
// --rfn of a real quant; a group's anchor is the rms of its members'), S_t fitted to the measured levels
// (geometric mean of kld / rfn_actual^alpha). Minimize the sum over targets subject to sum(numel * K) <=
// b * sum(numel) over the decoder linears (head, embedding and norms are outside the budget), K integer in
// [min-k, max-k]: start everything at min-k, raise the group with the largest KL drop per weight while the budget
// allows, then trade bits between groups while that lowers the predicted total (exchange repair). The defaults
// keep to what the B70 server runs (K 3-6, one K per fused group: q/k/v, gate/up). Output: exl3conv --recipe.
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

using json = nlohmann::json;

int main(int argc, char** argv)
{
    std::string mpath, apath, opath;
    double bits = 0, alpha = 2.0;
    int min_k = 3, max_k = 6;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(2); } return argv[++i]; };
        if (s == "-m") mpath = next(); else if (s == "-a") apath = next(); else if (s == "-o") opath = next();
        else if (s == "-b") bits = std::stod(next()); else if (s == "--alpha") alpha = std::stod(next());
        else if (s == "--min-k") min_k = std::stoi(next()); else if (s == "--max-k") max_k = std::stoi(next());
        else { std::fprintf(stderr, "usage: exl3opt -m noise.json -a rfn.json -b BITS -o recipe.json [--min-k 3] [--max-k 6] [--alpha 2.0]\n"); return 2; }
    }
    if (mpath.empty() || apath.empty() || opath.empty() || bits <= 0) { std::fprintf(stderr, "need -m, -a, -b and -o\n"); return 2; }
    try
    {
        const json mj = json::parse(std::ifstream(mpath)), aj = json::parse(std::ifstream(apath));
        std::map<std::string, std::pair<int, double>> anchor;
        for (const auto& r : aj.at("results")) anchor[r.at("key").get<std::string>()] = {r.at("K").get<int>(), r.at("rfn").get<double>()};

        struct G { std::string key; std::vector<std::string> members; double numel = 0, S = 0; int aK = 0; double arfn = 0; int K = 0; };
        std::vector<G> gs;
        for (const auto& r : mj.at("results"))
        {
            if (r.value("qbits_key", "bits") != "bits") continue;
            G g;
            g.key = r.at("key");
            g.members = r.contains("members") ? r.at("members").get<std::vector<std::string>>() : std::vector<std::string>{g.key};
            g.numel = r.at("numel").get<double>();
            // Anchor: the most common member K, rms of the members' rfn moved to it.
            std::map<int, int> kc;
            for (auto& mname : g.members) { auto it = anchor.find(mname); if (it == anchor.end()) throw std::runtime_error("no anchor for " + mname); ++kc[it->second.first]; }
            g.aK = std::max_element(kc.begin(), kc.end(), [](auto& x, auto& y) { return x.second < y.second; })->first;
            double sq = 0;
            for (auto& mname : g.members) { auto& an = anchor.at(mname); const double v = an.second * std::pow(1.96, an.first - g.aK); sq += v * v; }
            g.arfn = std::sqrt(sq / g.members.size());
            double ls = 0; int n = 0;
            for (const auto& l : r.at("levels"))
            {
                const double kld = l.at("kld").get<double>(), ra = l.at("rfn_actual").get<double>();
                if (kld > 0 && ra > 0) { ls += std::log(kld / std::pow(ra, alpha)); ++n; }
            }
            if (!n) throw std::runtime_error(g.key + ": no usable levels");
            g.S = std::exp(ls / n);
            gs.push_back(std::move(g));
        }
        auto rfn_at = [&](const G& g, int K) { return g.arfn * std::pow(1.96, g.aK - K); };
        auto kld = [&](const G& g, int K) { return g.S * std::pow(rfn_at(g, K), alpha); };
        double sum_numel = 0;
        for (auto& g : gs) sum_numel += g.numel;
        const double budget = std::floor(bits * sum_numel);
        double spent = min_k * sum_numel;
        if (spent > budget) throw std::runtime_error("min-k is above the budget");
        for (auto& g : gs) g.K = min_k;

        // Greedy: largest KL drop per weight first; ties to the earlier group.
        using E = std::tuple<double, size_t, int>;   // (-gain per weight, group, K after the step)
        std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
        for (size_t i = 0; i < gs.size(); ++i)
            if (min_k < max_k) heap.emplace(-(kld(gs[i], min_k) - kld(gs[i], min_k + 1)) / gs[i].numel, i, min_k + 1);
        while (!heap.empty())
        {
            auto [ng, i, kn] = heap.top(); heap.pop();
            if (spent + gs[i].numel > budget) continue;
            gs[i].K = kn; spent += gs[i].numel;
            if (kn < max_k) heap.emplace(-(kld(gs[i], kn) - kld(gs[i], kn + 1)) / gs[i].numel, i, kn + 1);
        }
        // Exchange repair: raise one group, paying with the cheapest demotions, while that lowers the total.
        for (int it = 0; it < 100; ++it)
        {
            const double left = budget - spent;
            struct C { double rate, loss, cost; size_t j; };
            std::vector<C> cands;
            for (size_t j = 0; j < gs.size(); ++j)
                if (gs[j].K > min_k) { const double loss = kld(gs[j], gs[j].K - 1) - kld(gs[j], gs[j].K); cands.push_back({loss / gs[j].numel, loss, gs[j].numel, j}); }
            std::sort(cands.begin(), cands.end(), [](const C& x, const C& y) { return x.rate < y.rate; });
            double best = 0; size_t bg = 0; std::vector<size_t> bdem;
            for (size_t g = 0; g < gs.size(); ++g)
            {
                if (gs[g].K >= max_k) continue;
                const double gain = kld(gs[g], gs[g].K) - kld(gs[g], gs[g].K + 1);
                double need = gs[g].numel - left, loss = 0;
                std::vector<size_t> dem;
                for (auto& c : cands)
                {
                    if (need <= 0) break;
                    if (c.j == g) continue;
                    dem.push_back(c.j); loss += c.loss; need -= c.cost;
                }
                if (need > 0) continue;
                if (gain - loss > best) { best = gain - loss; bg = g; bdem = dem; }
            }
            if (best <= 0) break;
            gs[bg].K += 1; spent += gs[bg].numel;
            for (auto j : bdem) { gs[j].K -= 1; spent -= gs[j].numel; }
        }

        double pred = 0, pred_uniform = 0;
        std::map<int, double> hist;
        json tensors = json::object();
        for (auto& g : gs)
        {
            pred += kld(g, g.K);
            pred_uniform += kld(g, (int) std::floor(bits));
            hist[g.K] += g.numel;
            for (auto& mname : g.members) tensors[mname] = g.K;
        }
        const double achieved = spent / sum_numel;
        json out = {{"source", "exl3opt"}, {"measurement", mpath}, {"anchors", apath}, {"target_bpw", bits},
                    {"achieved_bpw", std::round(achieved * 10000) / 10000}, {"alpha", alpha}, {"min_k", min_k}, {"max_k", max_k},
                    {"predicted_kld", pred}, {"bits", tensors}};
        std::ofstream(opath) << out.dump(1);
        std::printf("%zu groups, %.0f weights; achieved %.4f bpw (target %.2f)\n", gs.size(), sum_numel, achieved, bits);
        std::printf("predicted KL: %.5f (uniform %d bits: %.5f)\n", pred, (int) std::floor(bits), pred_uniform);
        for (auto& [k, w] : hist) std::printf("  K=%d: %5.1f%% of the weights\n", k, 100 * w / sum_numel);
        std::printf("-> %s\n", opath.c_str());
        return 0;
    }
    catch (const std::exception& ex) { std::fprintf(stderr, "error: %s\n", ex.what()); return 1; }
}
