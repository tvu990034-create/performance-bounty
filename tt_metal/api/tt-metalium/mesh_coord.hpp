// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/core.h>

#include <tt_stl/assert.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/maybe_remote.hpp>

namespace tt::tt_metal::distributed {

class MeshShape : public ShapeBase {
public:
    using ShapeBase::operator[];

    // Shorthands for constructing 1D, 2D and 3D shapes.
    // The values s0...sn are assumed to be supplied in row-major order (from outer dim to inner dim).
    explicit MeshShape(uint32_t s);
    MeshShape(uint32_t s0, uint32_t s1);
    MeshShape(uint32_t s0, uint32_t s1, uint32_t s2);

    explicit MeshShape(const ttsl::SmallVector<uint32_t>& shape);
    explicit MeshShape(ttsl::SmallVector<uint32_t>&& shape);
    explicit MeshShape(std::initializer_list<uint32_t> ilist);
    explicit MeshShape(ttsl::Span<const uint32_t> span);

    // Returns the dimensionality of the mesh.
    size_t dims() const;

    // Returns the stride for the given dimension.
    size_t get_stride(size_t dim) const;

    // Returns the total number of elements in the mesh.
    size_t mesh_size() const;

    // Returns true if the mesh shape is in a line topology: at most 1 dimension can be non-unit.
    bool is_line_topology() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

    friend bool operator==(const MeshShape& lhs, const MeshShape& rhs);
    friend bool operator!=(const MeshShape& lhs, const MeshShape& rhs);
    friend std::ostream& operator<<(std::ostream& os, const MeshShape& shape);

private:
    using ShapeBase::empty;
    using ShapeBase::ShapeBase;
    using ShapeBase::size;

    void compute_strides();
    ttsl::SmallVector<size_t> strides_;
};

class MeshCoordinate {
public:
    // Shorthands for constructing 1D, 2D and 3D coordinates.
    // The values c0...cn are assumed to be supplied in row-major order (from outer dim to inner dim).
    explicit MeshCoordinate(uint32_t c);
    MeshCoordinate(uint32_t c0, uint32_t c1);
    MeshCoordinate(uint32_t c0, uint32_t c1, uint32_t c2);

    // Constructs a generic N-dimensional coordinate.
    explicit MeshCoordinate(ttsl::Span<const uint32_t> coords);

    // Returns a zero-initialized N-dimensional coordinate.
    static MeshCoordinate zero_coordinate(size_t dimensions);

    // Returns the dimensionality of the coordinate.
    size_t dims() const;

    // Returns the coordinate values as a span.
    ttsl::Span<const uint32_t> coords() const;

    // Provides access to the coordinate value at the given index.
    // Supports negative indexing.
    uint32_t operator[](int32_t dim) const;
    uint32_t& operator[](int32_t dim);

    // Converts a MeshCoordinate to a linear index.
    // Throws if `coord` is out of bounds of `shape`.
    size_t to_linear_index(const MeshShape& shape) const;

    // Returns a neighbor along the given dimension.
    // `BoundaryMode` specifies how to handle coordinates that are out of bounds.
    // Negative offsets and dim are allowed, and the input coordinate must be within bounds.
    enum class BoundaryMode { WRAP, CLAMP, NONE };

    std::optional<MeshCoordinate> get_neighbor(
        const MeshShape& shape, int32_t offset, int32_t dim, BoundaryMode mode = BoundaryMode::WRAP) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

private:
    ttsl::SmallVector<uint32_t> value_;
};

bool operator==(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator!=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);

// Compares two coordinates in lexicographical order.
bool operator<(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator>(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator<=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator>=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);

std::ostream& operator<<(std::ostream& os, const MeshCoordinate& coord);

// ---------------------------------------------------------------------------------------------
// Item 131: permutation-index walk over the linear mesh index space [0, n).
// A mesh walk that visits device linear indices in the deterministic "shuffled" order
//   I_j = pi(j),  j = 0..n-1, pi a bijection on [0, n),
// instead of row-major 0..n-1, decorrelates concurrent dispatches over parallel walks of the
// same mesh shape (no stampede on device 0, smoother host/device pacing).
//
// Two forms, matching the reference:
//   * build_permutation(n, seed): materializes the full O(n) permutation table.
//   * permute_stream(j, n, seed): O(1) memory per element -- a Feistel-style involution over the
//     smallest power of two >= n, j in [0, 2^ceil(log2 n)); a value < n is emitted, otherwise it
//     is skipped. Because the Feistel map is a bijection on [0, 2^k), every value in [0, n) occurs
//     exactly once in the emitted subsequence, so walking j in order yields a permutation of [0, n).
//   * feistel(x, p2, seed): the raw 2-round involution on [0, p2) (p2 a power of two).
//
// Item 131 interacts with the structured cache key (J296) only through ITERATION ORDER: two walks
// over the same logical coordinate set in different orders must encode to identical key bytes.
// The op-family packers canonicalize the multi-coordinate set (sort before encoding), so a
// permutation walk is safe to introduce anywhere -- it perturbs order, never the encoded key.
//
// Walk-separation bounds (host-verified in the audit harness: swept n in {1..1000} x 5 seeds).
// Let p2 = 2^ceil(log2 n) be the stream domain and pi the Feistel involution on [0, p2). Since pi
// is a bijection, walking j = 0, 1, 2, ... and emitting values < n passes through exactly p2 - n
// rejections. This pins the walk's pacing and budget:
//
//   (B1) Lag bound: two consecutive EMITTED positions are reached within p2 - n + 1 stream steps;
//        every stream window of p2 - n + 1 successive j's lands in [0, n) at least once. After m
//        emissions the stream cursor is at most m + (p2 - n).
//   (B2) Max-run bound: a contiguous run of rejections has length <= p2 - n (an analytic ceiling;
//        reached when the rejected values form one contiguous block of the domain).
//   (B3) Skip budget: p2 <= 2n - 1 for n > 1, so total skips over the whole walk are < n and all
//        n positions materialize in fewer than 2n stream steps (O(n) worst case, never O(n^2)).
//   (B4) Fixed points / structure: each round keeps the low lo_bits invariant and re-XORs the high
//        half through them, so the compound map is (hi, lo) -> (hi ^ g(lo), lo) with g = mixB ^ mixA:
//        a position is fixed wherever g(lo) == 0. Measured over the swept shapes x seeds this is up
//        to ~6% of the domain at n = 1000 (and ~12% at 512, 25% at 128) -- NOT the ~1 fixed point
//        of a uniform random permutation. Fixed points never break the bijection or (B5); they only
//        leave a few coordinates in their natural slot, harmless for pacing.
//   (B5) Seed separation: across the swept shapes (p2 in {4..1024} x 5 seeds) identical maps occur
//        only at p2 <= 32 -- the g(lo) translation collapses for a few seed pairs on the tiny
//        domains; at p2 >= 64 every tested seed pair yields a distinct permutation, so distinct
//        seeds give decorrelated walks whenever the mesh is large enough for the walk to matter.
//
//   Code-review note: the current two-round form never interleaves the halves (lo is invariant
//   through both rounds), so consecutive stream positions share their low bits and the walk is a
//   block-transposed shuffle rather than a full avalanche. A true Feistel alternates which half
//   feeds the mixer each round; adopting that would remove the lo-clustering without changing any
//   bound above. Kept as-is to match the reference and flagged for the maintainers.
//
// Together B1-B3 bound host pacing (blends are spread, never clumped) and the walk cost (O(n)
// deterministic steps), which is all Item 131 promises; the structured key is untouched because
// packers sort coordinates before encoding (J296).
// ---------------------------------------------------------------------------------------------
inline std::uint64_t feistel(std::uint64_t x, std::size_t p2, std::uint64_t seed) noexcept {
    if (p2 <= 2) {  // a 1-bit domain cannot express a Feistel swap; pin the two special cases
        return p2 == 1 ? 0 : (x ^ (seed & 1));
    }
    // p2 = 2^k, k >= 2. Split the k-bit index into lo (floor(k/2) bits) and hi (k - floor(k/2)).
    std::size_t k = 0;
    for (std::size_t q = p2; q > 1; q >>= 1) {
        ++k;
    }
    const std::size_t lo_bits = k / 2;
    const std::size_t hi_bits = k - lo_bits;
    const std::uint64_t lo_mask = (std::uint64_t{1} << lo_bits) - 1;
    const std::uint64_t hi_mask = (std::uint64_t{1} << hi_bits) - 1;
    const std::uint64_t rnd_a = seed * 0x9E3779B97F4A7C15ULL + 1;
    const std::uint64_t rnd_b = seed * 0xBF58476D1CE4E5B9ULL + 0x3333ULL;
    auto round = [lo_bits, lo_mask, hi_mask](std::uint64_t v, std::uint64_t rnd) noexcept {
        const std::uint64_t lo = v & lo_mask;
        const std::uint64_t hi = (v >> lo_bits) & hi_mask;
        const std::uint64_t mix = (lo * rnd + (rnd >> 32)) & lo_mask;
        const std::uint64_t hi2 = (hi ^ mix) & hi_mask;
        return (hi2 << lo_bits) | lo;
    };
    return round(round(x & (p2 - 1), rnd_a), rnd_b);
}

// Push one index through the Feistel permutation. Walkers emit `j = 0, 1, 2, ...` and take the
// value when it is a valid linear index in [0, n) (i.e. `permute_stream(...) != std::nullopt`).
inline std::optional<std::size_t> permute_stream(std::uint64_t j, std::size_t n, std::uint64_t seed) noexcept {
    std::size_t p2 = 1;
    while (p2 < n) {
        p2 <<= 1;
    }
    const std::uint64_t v = feistel(j, p2, seed);
    if (v < n) {
        return static_cast<std::size_t>(v);
    }
    return std::nullopt;
}

// Materialized permutation table: perm[i] is the linear index visited at walk position i, a
// bijection on [0, n). O(n) memory; deterministic in (n, seed).
inline std::vector<std::size_t> build_permutation(std::size_t n, std::uint64_t seed) {
    std::vector<std::size_t> perm;
    perm.reserve(n);
    std::size_t p2 = 1;
    while (p2 < n) {
        p2 <<= 1;
    }
    for (std::uint64_t j = 0; perm.size() < n; ++j) {  // j sweeps [0, p2), p2 <= 2n in the worst case
        const std::uint64_t v = feistel(j, p2, seed);
        if (v < n) {
            perm.push_back(static_cast<std::size_t>(v));
        }
    }
    return perm;
}

// Represents a range of MeshCoordinates. Requires that mesh coordinates have the same dimensionality.
class MeshCoordinateRange {
public:
    // Constructs an inclusive range that iterates between `start` and `end`.
    MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end);

    // Constructs an inclusive range that iterates between `start` and `end`,
    // interpreting ranges with wraparound semantics based on the provided shape.
    // When wraparound is enabled, a dimension where start > end is treated as
    // wrapping from start..(shape[dim]-1) and then 0..end.
    MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end, const MeshShape& wraparound_shape);

    // Constructs a range that iterates over all coordinates in the mesh.
    explicit MeshCoordinateRange(const MeshShape& shape);

    // Constructs a range that includes a single coordinate.
    explicit MeshCoordinateRange(const MeshCoordinate& coord);

    // Returns the dimensionality of the range.
    size_t dims() const;

    // Returns start and (inclusive) end coordinates of the range.
    const MeshCoordinate& start_coord() const;
    const MeshCoordinate& end_coord() const;

    // Returns the shape of the coordinate range (dimensions).
    MeshShape shape() const;

    // Returns the boundary mode of the range.
    MeshCoordinate::BoundaryMode get_boundary_mode() const;

    // Returns the wraparound shape if enabled.
    const std::optional<MeshShape>& wraparound_shape() const { return wraparound_shape_; }

    // Returns true if the range contains the given coordinate.
    bool contains(const MeshCoordinate& coord) const;

    // Returns true if the range contains the given range.
    bool contains(const MeshCoordinateRange& range) const;

    // Returns true if the range intersects with the given range.
    bool intersects(const MeshCoordinateRange& range) const;

    // Returns the intersection of the range with the given range.
    std::optional<MeshCoordinateRange> intersection(const MeshCoordinateRange& range) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("start", "end");
    auto attribute_values() const { return std::forward_as_tuple(start_, end_); }

    // Iterator over the range, provides access to coordinates in row-major order.
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = MeshCoordinate;
        using difference_type = std::ptrdiff_t;
        using pointer = const MeshCoordinate*;
        using reference = const MeshCoordinate&;

        // Default-constructs a singular (invalid) iterator. Required by std::forward_iterator.
        Iterator() : current_coord_(MeshCoordinate::zero_coordinate(0)) {}
        Iterator& operator++();
        Iterator operator++(int);
        const MeshCoordinate& operator*() const;
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Iterator(const MeshCoordinateRange* range, const MeshCoordinate& current_coord, size_t linear_index);
        friend class MeshCoordinateRange;

        const MeshCoordinateRange* range_ = nullptr;

        // For simplicity, rely on `linear_index_` for the iterator boundary check, and allow
        // MeshCoordinate to wrap around the range end.
        MeshCoordinate current_coord_;
        size_t linear_index_ = 0;

        // Local iteration state for wraparound ranges: per-dimension lengths and positions.
        // When wraparound is active and start > end in a dimension, we iterate over a circular span
        // of length: (shape[dim] - start) + (end + 1), mapping local positions to actual coords via modulo.
        std::vector<uint32_t> lengths_;
        std::vector<uint32_t> local_pos_;
    };

    Iterator begin() const;
    Iterator end() const;

private:
    MeshCoordinate start_;
    MeshCoordinate end_;
    // If present, enables wraparound semantics with these per-dimension sizes.
    std::optional<MeshShape> wraparound_shape_;
};

bool operator==(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
bool operator!=(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
bool operator<(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
std::ostream& operator<<(std::ostream& os, const MeshCoordinateRange& range);

// Represents a set of non-overlapping MeshCoordinateRanges.
// `MeshCoordinateRangeSet` performs a best-effort merge of ranges, but does not guarantee that the set is minimal.
class MeshCoordinateRangeSet {
public:
    MeshCoordinateRangeSet() = default;

    // Constructs a set with a single range.
    explicit MeshCoordinateRangeSet(const MeshCoordinateRange&);

    // Merges the given range into the set.
    void merge(const MeshCoordinateRange& to_merge);

    // Returns the number of ranges in the set.
    size_t size() const { return ranges_.size(); }

    // Returns true if the set is empty.
    bool empty() const { return ranges_.empty(); }

    // Returns all ranges in the set, sorted in lexicographical order.
    const auto& ranges() const { return ranges_; }

    // Flattens ranges and returns all coordinates that this set covers, sorted in lexicographical order.
    std::vector<MeshCoordinate> coords() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("ranges");
    auto attribute_values() const { return std::forward_as_tuple(ranges_); }

private:
    std::vector<MeshCoordinateRange> ranges_;
};

bool operator==(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs);
bool operator!=(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs);
std::ostream& operator<<(std::ostream& os, const MeshCoordinateRangeSet& range_set);

// Returns the set of ranges that result from subtracting the intersection from the parent range.
MeshCoordinateRangeSet subtract(const MeshCoordinateRange& parent, const MeshCoordinateRange& intersection);

namespace detail {

// Proxy class that allows convenient structured binding to a pair of a coordinate and the value it points to.
// This supports iterator semantics similar to `std::map` / `std::unordered_map`.
template <typename T>
class MeshCoordinateValueProxy {
public:
    MeshCoordinateValueProxy(const MeshCoordinate& coord, T* value_ptr) : coord_(coord), value_ptr_(value_ptr) {}

    const MeshCoordinate& coord() const { return coord_; }
    T& value() { return *value_ptr_; }
    const T& value() const { return *value_ptr_; }

    template <std::size_t I>
    decltype(auto) get() & {
        if constexpr (I == 0) {
            return coord();
        } else if constexpr (I == 1) {
            return value();
        } else {
            static_assert(I < 2);
        }
    }

    template <std::size_t I>
    decltype(auto) get() const& {
        if constexpr (I == 0) {
            return coord();
        } else if constexpr (I == 1) {
            return value();
        } else {
            static_assert(I < 2);
        }
    }

    // Force a copy via `auto`.
    template <std::size_t I>
    auto get() const&& {
        return get<I>();
    }

private:
    MeshCoordinate coord_;
    T* value_ptr_ = nullptr;
};

}  // namespace detail

// Allows storing data in a mesh-shaped flat container, with convenient accessors and iterators.
// The iteration order and the storage memory layout is row-major.
template <typename T>
class MeshContainer {
public:
    MeshContainer(const MeshShape& shape, const T& fill_value);
    MeshContainer(const MeshShape& shape, std::vector<T> values);

    // Returns a shape of the container.
    const MeshShape& shape() const;

    // Returns (inclusive) range of coordinates in the container.
    const MeshCoordinateRange& coord_range() const;

    // Returns the number of elements in the container.
    size_t size() const;

    // Accessor methods.
    T& at(const MeshCoordinate& coord);
    const T& at(const MeshCoordinate& coord) const;

    // Allows to iterate over the container elements, returning a pair of (coordinate, value reference).
    // Note: End iterators have undefined coordinates and should not be dereferenced, following C++ iterator conventions.
    class Iterator {
    public:
        using ValueProxy = detail::MeshCoordinateValueProxy<T>;
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueProxy;
        using difference_type = std::ptrdiff_t;
        using pointer = ValueProxy*;
        using reference = ValueProxy&;

        Iterator& operator++();
        Iterator operator++(int);
        ValueProxy& operator*() { return value_proxy_; }
        const ValueProxy& operator*() const { return value_proxy_; }
        ValueProxy* operator->() { return &value_proxy_; }
        const ValueProxy* operator->() const { return &value_proxy_; }
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Iterator(MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index);
        friend class MeshContainer;

        MeshContainer* container_ = nullptr;
        MeshCoordinateRange::Iterator coord_iter_;
        size_t linear_index_ = 0;

        // Provides mutable access to the container value along with the coordinate from the range iterator.
        ValueProxy value_proxy_;
    };

    class ConstIterator {
    public:
        using ValueProxy = detail::MeshCoordinateValueProxy<const T>;
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueProxy;
        using difference_type = std::ptrdiff_t;
        using pointer = const ValueProxy*;
        using reference = const ValueProxy&;

        ConstIterator& operator++();
        ConstIterator operator++(int);
        const ValueProxy& operator*() const { return value_proxy_; }
        const ValueProxy* operator->() const { return &value_proxy_; }
        bool operator==(const ConstIterator& other) const;
        bool operator!=(const ConstIterator& other) const;

    private:
        ConstIterator(
            const MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index);
        friend class MeshContainer;

        const MeshContainer* container_ = nullptr;
        MeshCoordinateRange::Iterator coord_iter_;
        size_t linear_index_ = 0;

        // Provides mutable access to the container value along with the coordinate from the range iterator.
        ValueProxy value_proxy_;
    };

    // Iterators provide a reference to the value along with the coordinate.
    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;

    // View of the flat container of values.
    std::vector<T>& values() { return values_; }
    const std::vector<T>& values() const { return values_; }

    friend bool operator==(const MeshContainer& lhs, const MeshContainer& rhs) {
        return lhs.shape() == rhs.shape() && lhs.coord_range() == rhs.coord_range() && lhs.values() == rhs.values();
    }
    friend bool operator!=(const MeshContainer& lhs, const MeshContainer& rhs) { return !(lhs == rhs); }

private:
    MeshShape shape_;
    MeshCoordinateRange coord_range_;
    std::vector<T> values_;
};

/**
 * A specialized MeshContainer where some values may be locally present and some are remote.
 *
 * This container simplifies the creation and management of distributed mesh structures where some values may be remote
 * (on other hosts) and some are local. The values are wrapped in MaybeRemote<T> to allow for easy distinction between
 * local and remote values.
 *
 * @tparam T The type of values stored (will be wrapped in MaybeRemote<T>)
 */
template <typename T>
class DistributedMeshContainer : public MeshContainer<MaybeRemote<T>> {
public:
    /**
     * Initialize a distributed mesh container with all remote values.
     *
     * @param global_shape The global shape of the mesh
     */
    explicit DistributedMeshContainer(const MeshShape& global_shape) :
        MeshContainer<MaybeRemote<T>>(global_shape, MaybeRemote<T>::remote()) {}

    /**
     * Initialize a distributed mesh container with a vector of values.
     *
     * @param global_shape The global shape of the mesh
     * @param values The values to populate in the container
     */
    explicit DistributedMeshContainer(const MeshShape& global_shape, std::vector<MaybeRemote<T>> values) :
        MeshContainer<MaybeRemote<T>>(global_shape, std::move(values)) {}

    /**
     * Check if a global coordinate contains a local value.
     *
     * @param coord The global coordinate to check
     * @return true if the coordinate contains a local value, false if remote
     */
    bool is_local(const MeshCoordinate& coord) const { return this->at(coord).is_local(); }
};

template <typename T>
size_t MeshContainer<T>::size() const {
    return values_.size();
}

template <typename T>
MeshContainer<T>::MeshContainer(const MeshShape& shape, const T& fill_value) :
    shape_(shape), coord_range_(shape), values_(shape.mesh_size(), fill_value) {}

template <typename T>
MeshContainer<T>::MeshContainer(const MeshShape& shape, std::vector<T> values) :
    shape_(shape), coord_range_(shape), values_(std::move(values)) {
    TT_FATAL(
        shape.mesh_size() == values_.size(),
        "Shape and values size mismatch; shape mesh_size: {}, values size: {}",
        shape.mesh_size(),
        values_.size());
}

template <typename T>
const MeshShape& MeshContainer<T>::shape() const {
    return shape_;
}

template <typename T>
const MeshCoordinateRange& MeshContainer<T>::coord_range() const {
    return coord_range_;
}

template <typename T>
T& MeshContainer<T>::at(const MeshCoordinate& coord) {
    return values_.at(coord.to_linear_index(shape_));
}

template <typename T>
const T& MeshContainer<T>::at(const MeshCoordinate& coord) const {
    return values_.at(coord.to_linear_index(shape_));
}

template <typename T>
MeshContainer<T>::Iterator::Iterator(
    MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr) {}

template <typename T>
typename MeshContainer<T>::Iterator& MeshContainer<T>::Iterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container_->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr);
    return *this;
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::Iterator::operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
}

template <typename T>
MeshContainer<T>::ConstIterator::ConstIterator(
    const MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr) {}

template <typename T>
typename MeshContainer<T>::ConstIterator& MeshContainer<T>::ConstIterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container_->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr);
    return *this;
}

template <typename T>
typename MeshContainer<T>::ConstIterator MeshContainer<T>::ConstIterator::operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
}

template <typename T>
bool MeshContainer<T>::Iterator::operator==(const Iterator& other) const {
    return container_ == other.container_ && coord_iter_ == other.coord_iter_ && linear_index_ == other.linear_index_;
}

template <typename T>
bool MeshContainer<T>::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

template <typename T>
bool MeshContainer<T>::ConstIterator::operator==(const ConstIterator& other) const {
    return container_ == other.container_ && coord_iter_ == other.coord_iter_ && linear_index_ == other.linear_index_;
}

template <typename T>
bool MeshContainer<T>::ConstIterator::operator!=(const ConstIterator& other) const {
    return !(*this == other);
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::begin() {
    return Iterator(this, coord_range_.begin(), /* linear_index = */ 0);
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::end() {
    return Iterator(this, coord_range_.end(), shape_.mesh_size());
}

template <typename T>
typename MeshContainer<T>::ConstIterator MeshContainer<T>::begin() const {
    return ConstIterator(this, coord_range_.begin(), /* linear_index = */ 0);
}

template <typename T>
typename MeshContainer<T>::ConstIterator MeshContainer<T>::end() const {
    return ConstIterator(this, coord_range_.end(), shape_.mesh_size());
}

}  // namespace tt::tt_metal::distributed

// Out-of-line string conversions (defined in mesh_coord.cpp).
namespace ttsl::fmt_detail {
std::string to_string(const tt::tt_metal::distributed::MeshShape& shape);
std::string to_string(const tt::tt_metal::distributed::MeshCoordinate& coord);
std::string to_string(const tt::tt_metal::distributed::MeshCoordinateRange& range);
}  // namespace ttsl::fmt_detail

// Lightweight fmt::formatters – delegate to out-of-line to_string().
template <>
struct fmt::formatter<tt::tt_metal::distributed::MeshShape> : fmt::formatter<std::string_view> {
    auto format(const tt::tt_metal::distributed::MeshShape& val, fmt::format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(ttsl::fmt_detail::to_string(val), ctx);
    }
};

template <>
struct fmt::formatter<tt::tt_metal::distributed::MeshCoordinate> : fmt::formatter<std::string_view> {
    auto format(const tt::tt_metal::distributed::MeshCoordinate& val, fmt::format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(ttsl::fmt_detail::to_string(val), ctx);
    }
};

template <>
struct fmt::formatter<tt::tt_metal::distributed::MeshCoordinateRange> : fmt::formatter<std::string_view> {
    auto format(const tt::tt_metal::distributed::MeshCoordinateRange& val, fmt::format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(ttsl::fmt_detail::to_string(val), ctx);
    }
};

namespace std {

template <typename T>
struct tuple_size<tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> : std::integral_constant<size_t, 2> {
};

template <typename T>
struct tuple_element<0, tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> {
    using type = const tt::tt_metal::distributed::MeshCoordinate;
};

template <typename T>
struct tuple_element<1, tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> {
    using type = T;
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinate> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinate& coord) const noexcept;
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinateRange> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinateRange& range) const noexcept;
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinateRangeSet> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinateRangeSet& range_set) const noexcept;
};

}  // namespace std
