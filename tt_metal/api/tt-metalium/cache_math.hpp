// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ----------------------------------------------------------------------------------------------
// Canonical program-cache geometry and cost calculus. The equations here are the single source of
// truth for the cache tiers' sizing, probe expectations, placement mixers and cost models; the
// definitions in program_cache.hpp re-use this header's forms (same namespace, same names), and
// the audit harness cross-checks that nothing has drifted. The header is intentionally
// dependency-free (only the standard library), so it can be included from sizing tools, gtests and
// the op-family layer alike.
//
// Linear probing, Knuth bounds. Expected probes at load factor alpha:
//   succ   = (1 + 1/(1 - alpha)) / 2
//   unsucc = (1 + 1/(1 - alpha)^2) / 2          (an insert is an unsuccessful search)
// For a probe budget P_max the load factor must satisfy
//   alpha <= 1 - 1/sqrt(2*P_max - 1)
// so the tier capacity for at most nu_max resident tags is
//   capacity = round_up_power_of_two(ceil(nu_max / alpha_bound)).
// Reserving that capacity up front makes the ramp-up rehash-free; without a reserve, doubling
// growth costs exactly (C_max - C_0) * per_element, at most per_element / alpha amortized.
// ----------------------------------------------------------------------------------------------

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>

namespace tt::tt_metal::program_cache::math {

// Golden-ratio constant shared by every multiply-fold in the cache hash family (H231 combiner,
// feistel round mixes, sub-hash composition).
inline constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

// Probe budget for the tag tier: expected probes stay at or below this while the tier is at or
// under its reserved capacity (load factor ~0.42).
inline constexpr double kDefaultProbeBudget = 2.0;

// Payload tier grows at this load factor, deliberately looser than the tag tier's 0.5: a payload
// slot is a wide entry (96 B, tail-dominant), so fewer doubling steps buy real memory. The
// probe-cost increase is bounded by the Knuth unsuccessful-search bound (eta 0.7 -> 6.06 expected
// unsucc probes vs 2.5 at 0.5), and a hit still costs ~1.7 probes.
inline constexpr double kPayloadMaxLoadFactor = 0.7;

inline double expected_successful_probes(double alpha) noexcept {
    return (1.0 + 1.0 / (1.0 - alpha)) / 2.0;
}

inline double expected_unsuccessful_probes(double alpha) noexcept {
    const double one_minus_alpha = 1.0 - alpha;
    return (1.0 + 1.0 / (one_minus_alpha * one_minus_alpha)) / 2.0;
}

inline double alpha_bound_for_probe_bound(double probe_budget) noexcept {
    return 1.0 - 1.0 / std::sqrt(2.0 * probe_budget - 1.0);
}

// Birthday bound: with n keys drawn uniformly over a hash space of `space` buckets, the expected
// number of colliding pairs is C(n, 2) / space = n(n - 1) / (2 * space). For the 64-bit dispatch
// tag this is n(n - 1) / 2^65, the G201 soundness assumption: a collision only degrades to a
// byte-exact rebuild miss, never a wrong hit. At n = 1M the expected count is ~2.7e-8 -- the box
// is effectively collision-free at any realistic size.
inline double expected_collision_count(std::uint64_t n, std::uint64_t space) noexcept {
    return (static_cast<double>(n) * static_cast<double>(n - 1)) / (2.0 * static_cast<double>(space));
}

inline std::size_t round_up_power_of_two(std::size_t n) noexcept {
    std::size_t cap = 1;
    while (cap < n) {
        cap <<= 1;
    }
    return cap;
}

inline std::size_t capacity_for_distinct_keys(std::size_t nu_max, double probe_budget) noexcept {
    const double alpha = alpha_bound_for_probe_bound(probe_budget);
    const double needed = std::ceil(static_cast<double>(nu_max) / alpha);
    return round_up_power_of_two(static_cast<std::size_t>(needed));
}

// Capacity for up to `nu_max` keys at a target load factor (used by the payload tier's reserve,
// which shares the growth policy kPayloadMaxLoadFactor): ceil(nu_max / alpha), pow2-rounded.
inline std::size_t capacity_for_distinct_keys_at_load(std::size_t nu_max, double alpha) noexcept {
    const double needed = std::ceil(static_cast<double>(nu_max) / alpha);
    return round_up_power_of_two(static_cast<std::size_t>(needed));
}

// H231: incremental mixing fold over a byte span. Every 8-byte word joins the running hash with a
// multiply-rotate step (h ^= word; h = rotl(h, 27) * golden), the same golden-ratio/multiply
// family `ttsl::hash` uses for its combiner (G187). One definition, reused by every consumer --
// the structured tier's key fold and the op-family packers' composed sub-hashes (H232) -- so the
// equation lives in one file and is shared, not re-inlined per consumer.
inline std::uint64_t fold_bytes(const std::byte* p, std::size_t n_bytes, std::uint64_t seed = kGolden) noexcept {
    std::uint64_t h = seed;
    std::size_t off = 0;
    for (; off + sizeof(std::uint64_t) <= n_bytes; off += sizeof(std::uint64_t)) {
        std::uint64_t v;
        std::memcpy(&v, p + off, sizeof(v));
        h ^= v;
        h = std::rotl(h, 27);
        h *= kGolden;
    }
    for (; off < n_bytes; ++off) {
        const std::uint64_t b = static_cast<std::uint8_t>(p[off]);
        h ^= b;
        h = std::rotl(h, 27);
        h *= kGolden;
    }
    return h;
}

// Splitmix64 finalizer (G205). Program-cache hashes already end in a splitmix64 finalizer, so
// applying it a second time for placement only decorrelates the slot index from the payload map's
// bucket selection -- never adds entropy beyond the input's.
inline std::uint64_t avalanche(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// H232: deterministic sub-hash of tail/body material, folded with H231 and mixed, then XOR-ed into
// the discriminating head field of a structured key. Independent of head fields by construction so
// the family id alone never decides a key pair (B49 requirement).
inline std::uint64_t compose_sub_hash(const std::byte* body, std::size_t body_bytes, std::uint64_t sub_seed) noexcept {
    return avalanche(fold_bytes(body, body_bytes, sub_seed));
}

// Cache lines a probe run touches. The first slot of a run starting at bucket `bucket_index` lands
// at byte (bucket_index * slot_bytes) % line_bytes and the run then advances linearly, so the
// touched-line count is the ceiling of (offset + n_probes * slot_bytes) / line_bytes.
inline std::size_t lines_touched(
    std::size_t bucket_index, std::size_t n_probes, std::size_t slot_bytes, std::size_t line_bytes) noexcept {
    if (n_probes == 0) {
        return 0;
    }
    const std::size_t offset = (bucket_index * slot_bytes) % line_bytes;
    return (offset + n_probes * slot_bytes + line_bytes - 1) / line_bytes;
}

// Exact aggregate rehash work when a table grows by doubling from `initial_capacity` to
// `final_capacity`: every growth step re-inserts all live entries, and those copies form a
// geometric progression summing to (C_max - C_0) * copy_per_entry. Amortized over the entries live
// at the final table (n = alpha * C_max) that is (C_max - C_0)/(alpha * C_max) * c_copy, which
// never exceeds copy_per_entry / alpha.
inline double rehash_total(std::size_t initial_capacity, std::size_t final_capacity, double copy_per_entry) noexcept {
    return static_cast<double>(final_capacity - initial_capacity) * copy_per_entry;
}

inline double rehash_amortized(
    std::size_t initial_capacity,
    std::size_t final_capacity,
    double alpha,
    double copy_per_entry) noexcept {
    if (final_capacity == 0 || alpha <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(final_capacity - initial_capacity) /
           (alpha * static_cast<double>(final_capacity)) * copy_per_entry;
}

// Share of total dispatch cost spent inserting. Hits dominate at steady state, so a fraction below
// the hit-path noise floor means the insert path is not where dispatch latency lives.
inline double insertion_fraction(
    std::size_t total_dispatches, std::size_t inserts, double t_hit, double t_insert) noexcept {
    const double c_insert = static_cast<double>(inserts) * t_insert;
    const double c_hits = static_cast<double>(total_dispatches - inserts) * t_hit;
    const double total = c_hits + c_insert;
    return total > 0.0 ? c_insert / total : 0.0;
}

// Reporting helpers: insertion is a rate (1/time) for throughput, misses are a dimensionless
// fraction of dispatches, and at steady state (inserts a small share of dispatches) total dispatch
// cost is n_dispatches * t_hit.
inline double insert_rate(std::uint64_t n_inserts, double seconds) noexcept {
    return seconds > 0.0 ? static_cast<double>(n_inserts) / seconds : 0.0;
}

inline double miss_rate(std::uint64_t n_misses, std::uint64_t n_dispatches) noexcept {
    return n_dispatches > 0 ? static_cast<double>(n_misses) / static_cast<double>(n_dispatches) : 0.0;
}

inline double total_hot_path_cost(std::uint64_t n_dispatches, double t_hit) noexcept {
    return static_cast<double>(n_dispatches) * t_hit;
}

// Insert-path cost model. Every coefficient defaults to zero: they are meant to be fitted from a
// Tracy-separated measurement, never baked in here.
struct InsertCostModel {
    double hash_overhead = 0.0; // per-call hash overhead
    double hash_per_byte = 0.0; // hash cost per key byte
    double cmp_overhead = 0.0;  // compare per-call overhead
    double cmp_per_byte = 0.0;  // compare cost per key byte
    double probe_cost = 0.0;    // per-probe cache-line fetch
    double node_alloc = 0.0;    // node allocation, chained maps
    double malloc_cost = 0.0;   // key malloc, constant within a size class
    double write_overhead = 0.0;
    double write_per_byte = 0.0;
    double publish_cost = 0.0;
};

inline double predict_string_insert(const InsertCostModel& m, std::size_t key_bytes, double e_probes) noexcept {
    const double ell = static_cast<double>(key_bytes);
    return m.hash_overhead + m.hash_per_byte * ell                 // tau_hash
         + e_probes * (m.cmp_overhead + m.cmp_per_byte * ell)      // tau_compare
         + e_probes * m.probe_cost                                 // tau_probe
         + m.node_alloc + m.malloc_cost                            // tau_node_alloc + tau_malloc
         + m.write_overhead + m.write_per_byte * ell               // tau_write
         + m.publish_cost;                                         // tau_pub
}

inline double predict_pod_insert(const InsertCostModel& m, std::size_t pod_bytes, double e_probes) noexcept {
    const double ell = static_cast<double>(pod_bytes);
    return m.hash_overhead + m.hash_per_byte * ell
         + e_probes * (m.cmp_overhead + m.cmp_per_byte * ell)
         + e_probes * m.probe_cost
         + m.write_overhead + m.write_per_byte * ell
         + m.publish_cost;
}

// Savings of a fixed-width key migration for a fitted model. For equal byte counts the per-byte
// terms cancel and the saving reduces to the node and malloc terms that an open-addressed POD
// table removes.
inline double predicted_pod_savings(
    const InsertCostModel& m, std::size_t string_bytes, std::size_t pod_bytes, double e_probes) noexcept {
    return predict_string_insert(m, string_bytes, e_probes) - predict_pod_insert(m, pod_bytes, e_probes);
}

// Item 26: a key-representation migration phi is migration-safe only if the new hash agrees with
// the old one on every key the old cache already holds (h1(k) == h2(phi(k)) for all resident k).
// A sample test is necessary but not sufficient: any disagreement must abort the migration and
// rehash instead.
template <typename ForwardIterator, typename HashOne, typename HashTwo, typename Transform>
bool hash_migration_safe(ForwardIterator first, ForwardIterator last, HashOne h1, HashTwo h2, Transform phi) {
    for (; first != last; ++first) {
        if (h1(*first) != h2(phi(*first))) {
            return false;
        }
    }
    return true;
}

}  // namespace tt::tt_metal::program_cache::math