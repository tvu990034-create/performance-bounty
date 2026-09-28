// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt_stl/unique_any.hpp>
#include <tt_stl/overloaded.hpp>

#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::program_cache::detail {

template <typename shared_variables_t>
struct CachedProgram {
private:
    std::optional<tt::tt_metal::Program> owned_program;
    std::optional<shared_variables_t> owned_shared_variables;

    // Hidden to avoid misuse of `CachedProgram` as a proxy.
    CachedProgram(tt::tt_metal::Program& program, shared_variables_t& shared_variables) :
        program{program}, shared_variables{shared_variables} {}

public:
    tt::tt_metal::Program& program;

    // Cached program needs to share shared_variables between create and override_runtime_arguments functions
    shared_variables_t& shared_variables;

    CachedProgram(tt::tt_metal::Program&& program, shared_variables_t&& shared_variables) :
        owned_program{std::move(program)},
        owned_shared_variables{std::move(shared_variables)},
        program{*owned_program},
        shared_variables{*owned_shared_variables} {}

    // Move constructor is needed to make `CachedProgram` work with `unique_any`.
    // `CachedProgram` with owned `program` and `shared_variables` are moved; unowned proxies have their references
    // effectively copied.
    CachedProgram(CachedProgram&& other) noexcept :
        owned_program(std::move(other.owned_program)),
        owned_shared_variables(std::move(other.owned_shared_variables)),
        program(owned_program ? *owned_program : other.program),
        shared_variables(owned_shared_variables ? *owned_shared_variables : other.shared_variables) {}

    // Creates a "proxy" `CachedProgram` that references `program` and `shared_variables`.
    // Used for adapting `CachedMeshWorkload` to `CachedProgram`, and interfacing with TTNN Ops in
    // `override_runtime_arguments` methods.
    static CachedProgram proxy(tt::tt_metal::Program& program, shared_variables_t& shared_variables) {
        return CachedProgram{program, shared_variables};
    }

    CachedProgram(const CachedProgram&) = delete;
    CachedProgram& operator=(const CachedProgram&) = delete;
    CachedProgram& operator=(CachedProgram&&) = delete;
};

template <typename shared_variables_t>
struct CachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    shared_variables_t shared_variables;

    CachedMeshWorkload(tt::tt_metal::distributed::MeshWorkload&& workload, shared_variables_t&& shared_variables) :
        workload{std::move(workload)}, shared_variables{std::move(shared_variables)} {}
};

// Adapted cached mesh workload is used to interpop TT-distributed infra that dispatches MeshWorkloads and program
// factories written for a single device: programs and shared variables are created by single-device program factories,
// then stamped out to the entire mesh.
template <typename shared_variables_t>
struct AdaptedCachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<distributed::MeshCoordinateRange, shared_variables_t> shared_variables;

    AdaptedCachedMeshWorkload(
        tt::tt_metal::distributed::MeshWorkload&& workload,
        std::unordered_map<distributed::MeshCoordinateRange, shared_variables_t>&& shared_variables) :
        workload{std::move(workload)}, shared_variables{std::move(shared_variables)} {}
};

struct CachedProgramFactory {
    static constexpr auto MAX_SIZE = 4096;
    static constexpr auto ALIGNMENT = 32;

    ttsl::unique_any<MAX_SIZE, ALIGNMENT> cached_program;

    // Used to map a runtime value to a program factory type that is being used
    std::size_t program_factory_index = 0;

    template <typename shared_variables_t>
    CachedProgramFactory(CachedProgram<shared_variables_t>&& cached_program, std::size_t program_factory_index) :
        cached_program{std::move(cached_program)}, program_factory_index{program_factory_index} {}

    template <typename shared_variables_t>
    CachedProgramFactory(CachedMeshWorkload<shared_variables_t>&& cached_workload, std::size_t program_factory_index) :
        cached_program{std::move(cached_workload)}, program_factory_index{program_factory_index} {}

    template <typename shared_variables_t>
    CachedProgramFactory(
        AdaptedCachedMeshWorkload<shared_variables_t>&& cached_workload, std::size_t program_factory_index) :
        cached_program{std::move(cached_workload)}, program_factory_index{program_factory_index} {}
};

// Program-cache key: the 64-bit hash PLUS an exact, collision-free canonical encoding of the key
// material (see ttsl::hash::canonical_key). std::unordered_map uses the hash to pick a bucket and
// operator== to confirm the match -- so a 64-bit hash collision lands two distinct keys in the
// same bucket and is resolved by exact comparison instead of producing a wrong cache hit
// (issue #45821). `canonical` always carries at least an op-identity prefix, so distinct ops can
// never alias on a hash collision.
struct ProgramCacheKey {
    uint64_t hash = 0;
    std::string canonical;

    bool operator==(const ProgramCacheKey& other) const {
        return hash == other.hash && canonical == other.canonical;
    }

    // Custom operator<=> for strong ordering in collections that need it; std::unordered_map only needs operator== and
    // the hasher.
    std::strong_ordering operator<=>(const ProgramCacheKey& other) const {
        if (const auto cmp = hash <=> other.hash; cmp != std::strong_ordering::equal) {
            return cmp;
        }
        return canonical <=> other.canonical;
    }
};

// Custom hasher for ProgramCacheKey that uses the precomputed hash value.
// Other member(s) ignored since std::unordered_map will use operator== to resolve collisions within buckets.
struct ProgramCacheKeyHasher {
    std::size_t operator()(const ProgramCacheKey& key) const { return static_cast<std::size_t>(key.hash); }
};

// ----------------------------------------------------------------------------------------------
// Two-tier lookup: an open-addressed hash-tag table (the fast tier) gates access to the exact
// payload map (the slow tier).
//
// The tag tier is only ever an over-approximation: `contains(h)` is true iff some resident entry
// carries hash `h`, so a tag miss is a *definite* miss and skips canonical-string construction on
// the probe path, while a tag hit merely narrows the candidate set. Exactness is never traded
// away -- the payload tier still requires byte-for-byte canonical equality, so a 64-bit hash
// collision degrades to a rebuild miss instead of a wrong hit (issue #45821).
// ----------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------
// Geometry and cost calculus for the tiers (linear probing, Knuth bounds).
//
// Expected probes for linear probing at load factor alpha:
//   succ   = (1 + 1/(1 - alpha)) / 2
//   unsucc = (1 + 1/(1 - alpha)^2) / 2          (an insert is an unsuccessful search)
// For a probe budget P_max the load factor must satisfy
//   alpha <= 1 - 1/sqrt(2*P_max - 1)
// so the tier capacity for at most nu_max resident tags is
//   capacity = round_up_power_of_two(ceil(nu_max / alpha_bound)).
// Reserving that capacity up front makes the ramp-up rehash-free; without a
// reserve, doubling growth costs exactly (C_max - C_0) * per_element, at most
// per_element / alpha amortized.
//
// The insert cost model separates the measurable terms (hash, collision-driven
// compare, probe fetch, node allocation, key malloc, key write, publish). The
// coefficients are inputs to be fitted from a Tracy-separated measurement of a
// real cache; the model does not claim hardware constants of its own.
// ----------------------------------------------------------------------------------------------

// Probe budget for the tag tier: expected probes stay at or below this while
// the tier is at or under its reserved capacity (load factor ~0.42).
inline constexpr double kDefaultProbeBudget = 2.0;

// Payload tier grows at this load factor, deliberately looser than the tag tier's 0.5:
// a payload slot is a wide entry (96 B, tail-dominant), so fewer doubling steps buy real
// memory. The probe-cost increase is bounded by the Knuth unsuccessful-search bound
// (eta 0.7 -> 6.06 expected unsucc probes vs 2.5 at 0.5), and a hit still costs ~1.7 probes.
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

// Birthday bound: with n keys drawn uniformly over a hash space of `space` buckets, the
// expected number of colliding pairs is C(n, 2) / space = n(n - 1) / (2 * space). For the
// 64-bit dispatch tag this is n(n - 1) / 2^65, the G201 soundness assumption: a collision
// only degrades to a byte-exact rebuild miss, never a wrong hit. At n = 1M the expected
// count is ~2.7e-8 -- the box is effectively collision-free at any realistic size.
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
// multiply-rotate step (h ^= word; h = rotl(h, 27) * golden), the same golden-ratio/multiply family
// `ttsl::hash` uses for its combiner (G187). One definition, reused by every consumer -- the
// structured tier's key fold and the op-family packers' composed sub-hashes (H232) -- so the
// equation lives in one file and is shared, not re-inlined per consumer.
inline std::uint64_t fold_bytes(const std::byte* p, std::size_t n_bytes, std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept {
    std::uint64_t h = seed;
    std::size_t off = 0;
    for (; off + sizeof(std::uint64_t) <= n_bytes; off += sizeof(std::uint64_t)) {
        std::uint64_t v;
        std::memcpy(&v, p + off, sizeof(v));
        h ^= v;
        h = std::rotl(h, 27);
        h *= 0x9E3779B97F4A7C15ULL;
    }
    for (; off < n_bytes; ++off) {
        const std::uint64_t b = static_cast<std::uint8_t>(p[off]);
        h ^= b;
        h = std::rotl(h, 27);
        h *= 0x9E3779B97F4A7C15ULL;
    }
    return h;
}

// Cache lines a probe run touches. The first slot of a run starting at bucket
// `bucket_index` lands at byte (bucket_index * slot_bytes) % line_bytes and the
// run then advances linearly, so the touched-line count is the ceiling of
// (offset + n_probes * slot_bytes) / line_bytes.
inline std::size_t lines_touched(
    std::size_t bucket_index, std::size_t n_probes, std::size_t slot_bytes, std::size_t line_bytes) noexcept {
    if (n_probes == 0) {
        return 0;
    }
    const std::size_t offset = (bucket_index * slot_bytes) % line_bytes;
    return (offset + n_probes * slot_bytes + line_bytes - 1) / line_bytes;
}

// Exact aggregate rehash work when a table grows by doubling from
// `initial_capacity` to `final_capacity`: every growth step re-inserts all live
// entries, and those copies form a geometric progression summing to
// (C_max - C_0) * copy_per_entry. Amortized over the entries live at the final
// table (n = alpha * C_max) that is (C_max - C_0)/(alpha * C_max) * c_copy,
// which never exceeds copy_per_entry / alpha.
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

// Share of total dispatch cost spent inserting. Hits dominate at steady state,
// so a fraction below the hit-path noise floor means the insert path is not
// where dispatch latency lives.
inline double insertion_fraction(
    std::size_t total_dispatches, std::size_t inserts, double t_hit, double t_insert) noexcept {
    const double c_insert = static_cast<double>(inserts) * t_insert;
    const double c_hits = static_cast<double>(total_dispatches - inserts) * t_hit;
    const double total = c_hits + c_insert;
    return total > 0.0 ? c_insert / total : 0.0;
}

// Reporting helpers: insertion is a rate (1/time) for throughput, misses are a
// dimensionless fraction of dispatches, and at steady state (inserts a small
// share of dispatches) total dispatch cost is n_dispatches * t_hit.
inline double insert_rate(std::uint64_t n_inserts, double seconds) noexcept {
    return seconds > 0.0 ? static_cast<double>(n_inserts) / seconds : 0.0;
}

inline double miss_rate(std::uint64_t n_misses, std::uint64_t n_dispatches) noexcept {
    return n_dispatches > 0 ? static_cast<double>(n_misses) / static_cast<double>(n_dispatches) : 0.0;
}

inline double total_hot_path_cost(std::uint64_t n_dispatches, double t_hit) noexcept {
    return static_cast<double>(n_dispatches) * t_hit;
}

// Insert-path cost model. Every coefficient defaults to zero: they are meant to
// be fitted from a Tracy-separated measurement, never baked in here.
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

// Savings of a fixed-width key migration for a fitted model. For equal byte
// counts the per-byte terms cancel and the saving reduces to the node and
// malloc terms that an open-addressed POD table removes.
inline double predicted_pod_savings(
    const InsertCostModel& m, std::size_t string_bytes, std::size_t pod_bytes, double e_probes) noexcept {
    return predict_string_insert(m, string_bytes, e_probes) - predict_pod_insert(m, pod_bytes, e_probes);
}

// Item 26: a key-representation migration phi is migration-safe only if the new
// hash agrees with the old one on every key the old cache already holds
// (h1(k) == h2(phi(k)) for all resident k). A sample test is necessary but not
// sufficient: any disagreement must abort the migration and rehash instead.
template <typename ForwardIterator, typename HashOne, typename HashTwo, typename Transform>
bool hash_migration_safe(ForwardIterator first, ForwardIterator last, HashOne h1, HashTwo h2, Transform phi) {
    for (; first != last; ++first) {
        if (h1(*first) != h2(phi(*first))) {
            return false;
        }
    }
    return true;
}

// Flat open-addressed set of resident program-cache hashes, kept separate from the payload map so
// lookups only touch a cache-friendly uint64 array. The tag is the full 64-bit dispatch hash
// (H212/H215, tag-in-slot with t = 64) rather than a narrow 16/8-bit fingerprint: at realistic n
// the fp-gate saturates -- an 8-bit fingerprint over n uniform hashes is fully occupied with
// probability ~1 once n > ~695 (coupon-collector expectation sum_1..256(256/(256-i)) ~= 695), at
// which point every fresh miss pays the payload walk (measured 5.16 ns vs 0.68 ns gate skip), and
// a 16-bit fingerprint shows ~n/2^16 false candidates per lookup. A 64-bit tag keeps the false
// probability ~ n/2^64 (expected_collision_count, below), so the gate stays effectively exact.
// Set semantics also collapse the same-hash flood (G201 degenerate case): inserting the same tag
// N times inserts once and reports the short resident probe, so a key family that shares one
// dispatch hash never grows this tier's probe tail -- dedup-by-value, not dedup-by-count.
class HashTagSet {
public:
    // Pre-size the initial table to avoid a rehash on the first few inserts.
    static constexpr std::size_t kInitialCapacity = 64;

    // Insert-only, monotonic: a duplicate tag is a no-op. Grows by doubling at load factor 0.5,
    // never shrinks, and never leaves tombstones; rehash re-inserts every tag, preserving the
    // "contains" superset property.
    // Insert `hash` and return the number of tag-tier probes (slot reads) spent. A duplicate
    // insert reports the probes it took to find the resident tag -- the measurement hook for the
    // ramp-up geometry.
    std::size_t insert(std::uint64_t hash) {
        if (slots_.empty() || (count_ + 1) * 2 > slots_.size()) {
            grow();
        }
        const std::uint64_t tag = hash != 0 ? hash : 1;
        std::size_t probes = 0;
        for (std::size_t slot = start(tag);; slot = (slot + 1) & (slots_.size() - 1)) {
            ++probes;
            if (slots_[slot] == 0) {
                slots_[slot] = tag;
                ++count_;
                return probes;
            }
            if (slots_[slot] == tag) {
                return probes;  // set semantics: the tag is already resident
            }
        }
    }

    // Pre-size the tier for up to `distinct_max` resident tags from the default
    // probe budget, so the dispatch ramp-up grows the table exactly once here
    // instead of rehashing as inserts arrive. Aggregate rehash work is then zero
    // (see rehash_total for the exact doubling cost when no reserve is made).
    void reserve(std::size_t distinct_max) {
        const std::size_t target = capacity_for_distinct_keys(distinct_max, kDefaultProbeBudget);
        if (target > slots_.size()) {
            slots_.assign(target, 0);
        }
    }

    double load_factor() const noexcept {
        return slots_.empty() ? 0.0 : static_cast<double>(count_) / static_cast<double>(slots_.size());
    }

    // Read-only; never mutates the tier.
    bool contains(std::uint64_t hash) const noexcept {
        if (slots_.empty()) {
            return false;
        }
        const std::uint64_t tag = hash != 0 ? hash : 1;
        for (std::size_t slot = start(tag); slots_[slot] != 0; slot = (slot + 1) & (slots_.size() - 1)) {
            if (slots_[slot] == tag) {
                return true;
            }
        }
        return false;
    }

    void clear() noexcept {
        std::fill(slots_.begin(), slots_.end(), 0);
        count_ = 0;
    }

    std::size_t num_tags() const noexcept { return count_; }

    // Splitmix64 finalizer. Public because the structured tier uses the same mixer for its
    // placement (`StructuredProgramCache::mix`) and the gtests synthesize dispatch hashes with it.
    static std::uint64_t avalanche(std::uint64_t x) noexcept {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

private:
    // Power-of-two capacity with index = avalanche(tag) & (capacity - 1). Program-cache hashes
    // already end in a splitmix64 finalizer, so this extra fold only decorrelates placement here
    // from the payload map's bucket selection.
    std::size_t start(std::uint64_t tag) const noexcept { return mask(avalanche(tag)); }

    std::size_t mask(std::uint64_t x) const noexcept { return static_cast<std::size_t>(x) & (slots_.size() - 1); }

    void grow() {
        if (slots_.empty()) {
            slots_.assign(kInitialCapacity, 0);
            return;
        }
        std::vector<std::uint64_t> next(slots_.size() * 2, 0);
        const std::size_t next_mask = next.size() - 1;
        for (const std::uint64_t tag : slots_) {
            if (tag == 0) {
                continue;
            }
            std::size_t slot = static_cast<std::size_t>(avalanche(tag)) & next_mask;
            while (next[slot] != 0) {
                slot = (slot + 1) & next_mask;
            }
            next[slot] = tag;
        }
        slots_.swap(next);
    }

    std::vector<std::uint64_t> slots_;
    std::size_t count_ = 0;
};

// ----------------------------------------------------------------------------------------------
// Structured tier: a 64-byte payload-exact key and its open-addressed store.
//
// The string tier keys by {hash, canonical} and must rebuild the canonical string on every tag
// hit, because find() needs a complete key value. The structured tier replaces that material
// with a fixed-width POD whose exact payload is produced field-by-field by an op-family packer
// in the op layer, so a hit is one probe run of 64-byte compares with zero string or allocation
// work. Equality is byte-exact over the raw key bytes; the key format is versioned, so a field
// layout change bumps the version and old vs new keys are byte-unequal -- an exact miss that
// rebuilds, never a wrong hit (issue #45821). The tier is insert-only and monotonic: a duplicate
// key is a no-op, entries never shrink except through clear(), capacity only doubles, and rehash
// re-probes stored keys from their own bytes, so migration is exact by construction.
// ----------------------------------------------------------------------------------------------

// Key-format version carried in byte 0 of every structured key. Bump when any packer's field
// layout changes: keys built under the old layout are then byte-unequal to new ones, degrading
// to an exact (rebuild) miss instead of a stale hit. v2: op families now compose an
// H232 sub-hash into the u64 field at head offset 16 (previously reserved).
inline constexpr std::uint16_t kProgramCacheStructuredKeyVersion = 2;

// J271: the v2 wire format's head field offsets live HERE, as data, and nowhere in the op-family
// packers. Packers read this table instead of hardcoding byte offsets, so a layout change touches
// one table plus a version bump (old vs new keys are byte-unequal by construction, degrading to an
// exact miss), never packing code. This is the "offsets as data" equation: the field-by-field
// encoder is driven by the table, not by literals.
struct StructuredHeadLayout {
    std::uint16_t version_offset;      // u16@0  -- key-format version (byte 0 carries the version)
    std::uint16_t family_offset;       // u16@2  -- interned op-family id (registry tag)
    std::uint16_t sub_offset;          // u64@16 -- H232 sub-hash discriminator (a deterministic fold
                                       //           of the tail material, independent of head fields)
    std::uint16_t coord_count_offset;  // u8@24  -- 1 single coord in head; 0 none; 0xFF coords in tail
    std::uint16_t coord_x_offset;      // u32@25 -- single-coordinate x (in-head fast path)
    std::uint16_t coord_y_offset;      // u32@29 -- single-coordinate y (in-head fast path)
    std::uint16_t head_bytes;          // 64     -- one cache line, bounds the tier memcmp
};
inline constexpr StructuredHeadLayout kStructuredHeadV2{0, 2, 16, 24, 25, 29, 64};

// Part K / D97+B31: the future v3 head-only row. Offsets are the D97-computed packed layout
// (version@0, family@2, sub@4, coord_count@16, dtype@17, layout@18, coord_x@19, coord_y@24), a
// 32-byte head, no tail. The fast-path bucket derives from the 4-byte word at offset 16
// (coord_count|dtype|layout|x-lsb), so the hit path reads one word, probes, then does a single
// 32-byte memcmp. Gated off: flipping kUseStructuredHeadV3 routes families to the packed tier
// behind a version bump (id 3), and a stale v2 key is byte-unequal to v3 keys built from identical
// inputs -- an exact (rebuild) miss, never a wrong hit (issue #45821 rule).
inline constexpr bool kUseStructuredHeadV3 = false;
struct StructuredHeadLayoutV3 {
    std::uint16_t version_offset;      // u16@0
    std::uint16_t family_offset;       // u16@2
    std::uint16_t sub_offset;          // u64@4 -- H232 fold(exact, kSubSeed); no body in head-only
    std::uint16_t coord_count_offset;  // u8@16
    std::uint16_t dtype_offset;        // u8@17
    std::uint16_t layout_offset;       // u8@18
    std::uint16_t coord_x_offset;      // u32@19
    std::uint16_t coord_y_offset;      // u32@24
    std::uint16_t head_bytes;          // 32
    std::uint16_t fast_word_offset;    // 16
};
inline constexpr StructuredHeadLayoutV3 kStructuredHeadV3{0, 2, 4, 16, 17, 18, 19, 24, 32, 16};
// Part L finding: the v3 slot probe must compare via std::memcmp (as ProgramCacheStructuredKey::
// operator== already does). std::array<std::byte,32>::operator== lowers to a scalar equal-loop in
// libstdc++ and measured ~4x slower per probe in the host benchmark; memcmp keeps the head-only
// hit path at one cache-half compare.

// A 64-byte (one cache line) payload key. Opaque to the tier: the bytes are laid out by the
// owning op family and read/written through the typed load/store helpers, which are memcpy-based
// so alignment and strict-aliasing rules are never bent. Comparison is byte-exact.
class ProgramCacheStructuredKey {
public:
    static constexpr std::size_t kSizeBytes = 64;

    template <typename T>
    T load(std::size_t offset) const {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        std::memcpy(&value, bytes_ + offset, sizeof(T));
        return value;
    }

    template <typename T>
    void store(std::size_t offset, T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(bytes_ + offset, &value, sizeof(T));
    }

    const std::byte* data() const noexcept { return bytes_; }

    bool operator==(const ProgramCacheStructuredKey& other) const noexcept {
        return std::memcmp(bytes_, other.bytes_, kSizeBytes) == 0;
    }

    bool operator!=(const ProgramCacheStructuredKey& other) const noexcept { return !(*this == other); }

private:
    std::byte bytes_[kSizeBytes] = {};
};

static_assert(std::is_standard_layout_v<ProgramCacheStructuredKey>);
static_assert(sizeof(ProgramCacheStructuredKey) == ProgramCacheStructuredKey::kSizeBytes);
// The offsets-as-data table must stay inside the real key bounds (J271 wire-format guard): a
// packer that overruns the fixed head fails to compile instead of corrupting the byte-exact tier.
static_assert(kStructuredHeadV2.head_bytes == ProgramCacheStructuredKey::kSizeBytes);
static_assert(kStructuredHeadV2.coord_y_offset + 4 <= ProgramCacheStructuredKey::kSizeBytes);
// Part K guards: the v3 head-only tier is a 32-byte key, its fast word sits at offset 16, and no
// field may cross the head boundary. Off here, but the table must stay self-consistent.
static_assert(kStructuredHeadV3.head_bytes == 32);
static_assert(kStructuredHeadV3.fast_word_offset == 16);
static_assert(kStructuredHeadV3.coord_y_offset + 4 <= kStructuredHeadV3.head_bytes);

// Exact variable-length companion to the fixed head: the op layer appends a length-prefixed, byte
// exact encoding of the key material that does not fit in the 64-byte head (e.g. activation
// sequences, core-range sets). Two keys are equal only if BOTH the head bytes and the tail bytes
// match, so the tail can only refine (never relax) the head's equality -- the byte-exact guarantee
// of the structured tier holds over the (head, tail) pair.
using ProgramCacheStructuredTail = std::shared_ptr<const std::vector<std::byte>>;

// The all-default (nothing set) tail is one shared, immutable, empty block: the op layer returns it
// as-is whenever every optional field is at its default, so the common dispatch's probe does zero
// allocation. This is the F155 rule of the structured key spec (omit-all-defaults -> empty tail).
inline const ProgramCacheStructuredTail kEmptyProgramCacheStructuredTail =
    std::make_shared<const std::vector<std::byte>>();

// Result of an op family's structured key pack: the 64-byte head plus its exact variable-length
// tail (kEmptyProgramCacheStructuredTail when every optional field is at its default).
struct ProgramCacheStructuredPack {
    ProgramCacheStructuredKey head;
    ProgramCacheStructuredTail tail;
};

inline bool structured_key_bytes_equal(
    const ProgramCacheStructuredTail& a, const ProgramCacheStructuredTail& b) noexcept {
    if (a == b) {
        return true;  // shared (or both null) -> identical payload by construction (immutable)
    }
    if (!a || !b) {
        return false;
    }
    return *a == *b;
}

// Exact op-identity interner. The structured key carries op identity as a small interned id:
// distinct type names map to distinct ids, so two ops can never alias on a key's byte layout.
// `intern` is called once per op type (memoized in a function-local static by the op layer), so
// a per-dispatch probe does zero string work after the first call. 0 and 0xFFFF are reserved:
// intern returns 0 when the id space is exhausted, and a caller that gets 0 must fall back to
// the string tier.
struct TransparentStringHash {
    using is_transparent = void;
    // G187/H231: the registry folds short structured type names with the same golden-ratio mixing
    // the tier uses elsewhere (fold_bytes), instead of the seedless libstdc++ FNV on string_view,
    // which is fingerprint-flaky for the small, related spellings that dominate op type names.
    std::size_t operator()(std::string_view sv) const noexcept {
        return static_cast<std::size_t>(fold_bytes(reinterpret_cast<const std::byte*>(sv.data()), sv.size()));
    }
};

class StructuredOpRegistry {
public:
    static StructuredOpRegistry& instance() {
        static StructuredOpRegistry registry;
        return registry;
    }

    std::uint16_t intern(std::string_view type_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = ids_.find(type_name);
        if (it != ids_.end()) {
            return it->second;
        }
        if (next_id_ == 0xFFFF) {
            return 0;  // 65534 usable ids cannot be exhausted by a realistic process
        }
        const std::uint16_t id = next_id_++;
        ids_.emplace(type_name, id);
        return id;
    }

private:
    StructuredOpRegistry() = default;
    std::unordered_map<std::string, std::uint16_t, TransparentStringHash, std::equal_to<>> ids_;
    std::mutex mutex_;
    std::uint16_t next_id_ = 1;
};

// Flat open-addressed store of structured keys with their cached program factories inline.
//
// Two probe-cost safeguards over the base open-addressing scheme:
//  * G201 hash fast-reject: the slot's placement hash is compared before the 64-byte head memcmp,
//    so a probe that lands on a foreign entry costs one 8-byte compare instead of a 64-byte one.
//  * Key-byte placement: the probe start is a function of BOTH the dispatch hash and the exact key
//    bytes (G205 XOR-fold). A flood of distinct keys that share one dispatch hash therefore spreads
//    across the table instead of collapsing into a single probe cluster, so the worst-case probe
//    tail under hash collisions stays bounded by construction. The tag tier still gates on the
//    dispatch hash alone (a necessary condition), which is unaffected.
class StructuredProgramCache {
public:
    // Pre-size the initial table to avoid a rehash on the first few inserts.
    static constexpr std::size_t kInitialCapacity = 64;

    struct Slot {
        std::uint8_t state = 0;  // 0 = empty, 1 = occupied
        std::uint64_t hash = 0;  // dispatch hash the slot was entered under (see grow)
        ProgramCacheStructuredKey key;
        ProgramCacheStructuredTail tail;  // exact variable-length companion (see above)
        std::unique_ptr<CachedProgramFactory> factory;
    };

    // Insert-only, monotonic: a duplicate key is a no-op returning the resident entry, the entry
    // count never shrinks except through clear(), and capacity doubles at kPayloadMaxLoadFactor
    // (0.7) -- the tag tier keeps its own 0.5 rule, since tags are 8-byte words while these slots
    // are wide and memory-dominant -- without ever shrinking. grow() re-probes the stored slots
    // from their own bytes (dispatch hash + key), so the rehash is exact by construction; the
    // payload (unique_ptr) is never recopied. A duplicate never overwrites and registers no new tag.
    CachedProgramFactory* insert(
        std::uint64_t hash, ProgramCacheStructuredKey key, ProgramCacheStructuredTail tail, CachedProgramFactory&& factory) {
        if (slots_.empty() || static_cast<double>(count_ + 1) > static_cast<double>(slots_.size()) * kPayloadMaxLoadFactor) {
            grow();
        }
        for (std::size_t slot = start(hash, key);; slot = (slot + 1) & (slots_.size() - 1)) {
            if (slots_[slot].state == 0) {
                slots_[slot].state = 1;
                slots_[slot].hash = hash;
                slots_[slot].key = key;
                slots_[slot].tail = std::move(tail);
                slots_[slot].factory = std::make_unique<CachedProgramFactory>(std::move(factory));
                ++count_;
                tags_.insert(hash);
                return slots_[slot].factory.get();
            }
            if (slots_[slot].hash == hash && slots_[slot].key == key &&
                structured_key_bytes_equal(slots_[slot].tail, tail)) {
                return slots_[slot].factory.get();  // set semantics: already resident
            }
        }
    }

    // Tag-gated byte-exact probe: `hash` is the dispatch hash the launch path already computed. A
    // tag miss is a definite miss (one word probe, no slot-table touch); a tag hit only narrows the
    // candidate set, and the slot probe still demands byte-exact (head, tail) equality before
    // reporting a hit, so a mismatched key is never a wrong hit.
    CachedProgramFactory* lookup(
        std::uint64_t hash, const ProgramCacheStructuredKey& key, const ProgramCacheStructuredTail& tail) {
        if (!tags_.contains(hash)) {
            return nullptr;
        }
        for (std::size_t slot = start(hash, key); slots_[slot].state != 0;
             slot = (slot + 1) & (slots_.size() - 1)) {
            const Slot& s = slots_[slot];
            if (s.hash == hash && s.key == key && structured_key_bytes_equal(s.tail, tail)) {
                return s.factory.get();
            }
        }
        return nullptr;
    }

    // Pre-size for up to `distinct_max` resident keys at the payload growth policy
    // (`kPayloadMaxLoadFactor`), so the dispatch ramp-up grows the table exactly once here instead
    // of rehashing as inserts arrive. Aggregate rehash work is then zero (see rehash_total for the
    // exact doubling cost when no reserve is made).
    void reserve(std::size_t distinct_max) {
        const std::size_t target = capacity_for_distinct_keys_at_load(distinct_max, kPayloadMaxLoadFactor);
        if (target > slots_.size()) {
            slots_.resize(target);
        }
        tags_.reserve(distinct_max);
    }

    double load_factor() const noexcept {
        return slots_.empty() ? 0.0 : static_cast<double>(count_) / static_cast<double>(slots_.size());
    }

    std::size_t size() const noexcept { return count_; }
    std::size_t capacity() const noexcept { return slots_.size(); }

    void clear() noexcept {
        slots_.clear();
        slots_.resize(kInitialCapacity);
        tags_.clear();
        count_ = 0;
    }

private:
    // G205-style fold of the exact 64-byte head, applied as an H231 incremental mix
    // (`fold_bytes`): each word folds in with a multiply-rotate step, so every byte contributes to
    // placement and word-identical chunks decorrelate instead of XOR-canceling. Mixing this into
    // the probe start spreads distinct keys that share a dispatch hash across different clusters
    // (see the class comment). Deterministic per head bytes, so grow() re-derives placements exactly.
    static std::uint64_t fold_key(const ProgramCacheStructuredKey& key) noexcept {
        return fold_bytes(key.data(), ProgramCacheStructuredKey::kSizeBytes);
    }

    // Probe placement = the same splitmix avalanche HashTagSet applies to its tags, applied to the
    // dispatch hash mixed with the key bytes. Deterministic per (hash, key), so grow() recomputes
    // placements from the stored slots; the dispatch hash is stored per slot precisely because it
    // is external to the key bytes (G201 fast-reject).
    static std::uint64_t mix(std::uint64_t x) noexcept { return HashTagSet::avalanche(x); }

    std::size_t mask(std::uint64_t x) const noexcept { return static_cast<std::size_t>(x) & (slots_.size() - 1); }

    std::size_t start(std::uint64_t hash, const ProgramCacheStructuredKey& key) const noexcept {
        return mask(mix(hash ^ fold_key(key)));
    }

    void grow() {
        if (slots_.empty()) {
            slots_.resize(kInitialCapacity);
            return;
        }
        std::vector<Slot> next(slots_.size() * 2);
        const std::size_t next_mask = next.size() - 1;
        for (auto& slot : slots_) {
            if (slot.state == 0) {
                continue;
            }
            std::size_t target = mask(mix(slot.hash ^ fold_key(slot.key))) & next_mask;
            while (next[target].state != 0) {
                target = (target + 1) & next_mask;
            }
            next[target] = std::move(slot);
        }
        slots_.swap(next);
    }

    HashTagSet tags_;
    std::vector<Slot> slots_;
    std::size_t count_ = 0;
};

// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from
// TT-Metal and TT-Eager using ttsl::concepts::unique_any.
struct ProgramCache {
    // Two-tier lookup: a tag miss is a definite miss and skips the exact payload probe. A tag hit
    // only narrows the candidate set; the payload tier still demands byte-for-byte canonical
    // equality before reporting a hit.
    bool contains(const ProgramCacheKey& program_key) const {
        if (!tags_.contains(program_key.hash)) {
            return false;
        }
        return this->cache_.contains(program_key);
    }

    // Whether any resident entry carries this hash. True only marks a candidate; false is a
    // *definite* miss. The launch path calls this *before* building the canonical key string, so
    // a fresh-key probe is resolved without string construction. Never used to decide a hit on
    // its own -- callers must still confirm with `lookup`.
    bool has_hash(std::uint64_t hash) const noexcept { return tags_.contains(hash); }

    CachedProgramFactory& get(const ProgramCacheKey& program_key) { return this->cache_.at(program_key); }

    // Single fully-exact probe: returns the cached entry iff the exact key (hash + canonical
    // bytes) is resident, else nullptr. The steady-state hit path uses one map find instead of
    // the contains()+get() pair; a tag miss short-circuits without touching the payload map.
    CachedProgramFactory* lookup(const ProgramCacheKey& program_key) {
        if (!tags_.contains(program_key.hash)) {
            return nullptr;
        }
        auto it = this->cache_.find(program_key);
        return it != this->cache_.end() ? &it->second : nullptr;
    }

    // Insert `program` for `program_key`. The key is *moved* into the entry -- the canonical
    // string's buffer is adopted, not byte-copied -- which zeroes the key-write term of the
    // insert cost model on the first-use path. Returns the inserted-or-already-resident entry
    // so the caller needs no second find. Map keys are const, so moving requires a
    // piecewise-construct emplace. A duplicate insert never overwrites and registers no tag.
    CachedProgramFactory* insert(ProgramCacheKey&& program_key, CachedProgramFactory&& program) {
        auto [it, inserted] = this->cache_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::move(program_key)),
            std::forward_as_tuple(std::move(program)));
        if (inserted) {
            tags_.insert(it->first.hash);
        }
        return &it->second;
    }

    // Lvalue overload for callers that must keep the key after the call; the canonical string
    // is copied in that case.
    CachedProgramFactory* insert(const ProgramCacheKey& program_key, CachedProgramFactory&& program) {
        return insert(ProgramCacheKey(program_key), std::move(program));
    }

    // Structured tier: tag-gated byte-exact probe of the (head, tail) key. `hash` is the same
    // dispatch hash that gates the string tier, so a tag miss is a definite miss for BOTH tiers
    // and `has_hash` never reports a false negative for a resident structured key. The op layer
    // uses these only for ops whose packer can encode the full key material field-by-field; every
    // other op keeps the string tier.
    CachedProgramFactory* lookup_structured(
        std::uint64_t hash,
        const ProgramCacheStructuredKey& program_key,
        const ProgramCacheStructuredTail& program_tail) {
        return structured_.lookup(hash, program_key, program_tail);
    }

    CachedProgramFactory* insert_structured(
        std::uint64_t hash,
        ProgramCacheStructuredKey program_key,
        ProgramCacheStructuredTail program_tail,
        CachedProgramFactory&& program) {
        tags_.insert(hash);
        return structured_.insert(hash, program_key, std::move(program_tail), std::move(program));
    }

    // Pre-size both tiers for up to `distinct_max` resident keys so the ramp-up
    // makes no rehash at all. The payload map reserves `distinct_max` buckets
    // and the tag tier is sized from the default probe budget (expected probes
    // <= kDefaultProbeBudget). Reserving up front zeroes the aggregate rehash
    // work; doubling growth on an unreserved table costs exactly
    // rehash_total, which amortizes to at most copy_per_entry/alpha.
    void reserve(std::size_t distinct_max) {
        this->cache_.reserve(distinct_max);
        this->tags_.reserve(distinct_max);
        this->structured_.reserve(distinct_max);
    }

    // Verification hooks for the tier geometry: the tag tier's current load
    // factor and the expected probes of the next unsuccessful insert at that
    // load. Used to confirm the ramp-up stays within the probe budget.
    double tag_load_factor() const noexcept { return tags_.load_factor(); }
    double expected_insert_probes() const noexcept { return expected_unsuccessful_probes(tags_.load_factor()); }
    double structured_load_factor() const noexcept { return structured_.load_factor(); }

    void enable() { is_enabled_ = true; }

    void disable() { is_enabled_ = false; }

    bool is_enabled() const { return is_enabled_; }

    void set_cache_misses_allowed(bool allowed) { allow_cache_misses_ = allowed; }
    bool cache_misses_allowed() const { return allow_cache_misses_; }

    void clear() {
        this->cache_.clear();
        this->tags_.clear();
        this->structured_.clear();
    }

    // Distinct-key count across both tiers: monotone non-decreasing over the device's lifetime
    // until clear().
    std::size_t num_entries() const { return this->cache_.size() + structured_.size(); }

private:
    bool is_enabled_ = true;
    bool allow_cache_misses_ = true;
    HashTagSet tags_;
    std::unordered_map<ProgramCacheKey, CachedProgramFactory, ProgramCacheKeyHasher> cache_;
    StructuredProgramCache structured_;
};

}  // namespace tt::tt_metal::program_cache::detail
