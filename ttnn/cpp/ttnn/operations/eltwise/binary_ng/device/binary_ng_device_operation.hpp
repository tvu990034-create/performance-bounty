// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <cstdint>
#include <tuple>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/distributed/types.hpp"
#include <optional>
#include <vector>
namespace ttnn::operations::binary_ng {

enum class SubtileBroadcastType {
    NONE,         // both tensors have equal tile dimensions (H & W)
    SCALAR_A,     // a is a scalar (H = 1, W = 1)
    SCALAR_B,     // b is a scalar (H = 1, W = 1)
    ROW_A_COL_B,  // a has a single tile row, b has a single tile column
    ROW_B_COL_A,  // b has a single tile row, a has a single tile column
    ROW_A,        // a has a single tile row, b is full
    ROW_B,        // b has a single tile row, a is full
    COL_A,        // a has a single tile column, b is full
    COL_B,        // b has a single tile column, a is full
};

SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w);

struct BinaryNgDeviceOperation {
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> lhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> rhs_activations;
        ttsl::SmallVector<unary::EltwiseUnaryWithParam> post_activations;
        std::optional<unary::ScalarVariant> scalar;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        std::optional<DataType> dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        std::optional<CoreRangeSet> sub_core_grids;
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
        SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;
        // The scalar operand is the mathematical left-hand side, so the compute kernel
        // evaluates op(scalar, tensor). Only ever set on the scalar path, where the tensor
        // occupies slot a and the scalar slot b regardless of operand order.
        bool scalar_is_lhs = false;
        bool is_sfpu = false;
        bool is_quant_op = false;
        bool is_where_op = false;
        float rtol = 0.0f;
        float atol = 0.0f;
        bool equal_nan = false;
        Layout input_layout_a = Layout::TILE;
        Layout input_layout_b = Layout::TILE;
        Layout output_layout = Layout::TILE;
        std::optional<std::uint32_t> a_shard_volume;
        std::optional<std::uint32_t> b_shard_volume;
        std::optional<std::uint32_t> c_shard_volume;
        // Sharded output's shape in pages on the accessor path. The inputs' equivalent rides in
        // tensor_args_t::to_hash(); the output has no Tensor at hash time, so it is carried here.
        std::optional<tt::tt_metal::Shape> c_tensor_shape_in_pages;

        DataType get_dtype() const;

        // Program-cache attributes. Runtime scalar/tolerance values are excluded; they are patched as runtime args.
        static constexpr auto attribute_names = std::make_tuple(
            "binary_op_type",
            "lhs_activations",
            "rhs_activations",
            "post_activations",
            "memory_config",
            "dtype",
            "compute_kernel_config",
            "sub_core_grids",
            // core_ranges of every CB and kernel. Depends on the device's sub-device layout, which
            // nothing else here carries and which does not clear the cache when swapped. Same set on a
            // single sub-device, so no extra entries there.
            "worker_grid",
            "subtile_broadcast_type",
            "scalar_is_lhs",
            "is_sfpu",
            "is_quant_op",
            "is_where_op",
            "input_layout_a",
            "input_layout_b",
            "output_layout",
            "equal_nan",
            "a_shard_volume",
            "b_shard_volume",
            "c_shard_volume",
            "c_tensor_shape_in_pages");

        auto attribute_values() const {
            return std::make_tuple(
                binary_op_type,
                lhs_activations,
                rhs_activations,
                (is_where_op || is_quant_op) ? ttnn::SmallVector<unary::EltwiseUnaryWithParam>{} : post_activations,
                memory_config,
                get_dtype(),
                compute_kernel_config,
                sub_core_grids,
                worker_grid,
                subtile_broadcast_type,
                scalar_is_lhs,
                is_sfpu,
                is_quant_op,
                is_where_op,
                input_layout_a,
                input_layout_b,
                output_layout,
                binary_op_type == BinaryOpType::ISCLOSE ? equal_nan : false,
                a_shard_volume,
                b_shard_volume,
                c_shard_volume,
                c_tensor_shape_in_pages);
        }
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;

        // Operand dtypes, memory configs, Alignment, and Tile, plus each sharded operand's shape in
        // pages. Omits logical shape by design, so differently-shaped interleaved calls share one cache entry.
        ttsl::hash::hash_t to_hash() const;
    };

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c);

        // Cache-hit re-apply of all per-dispatch state (per-core args + tensor-backed CB/buffer
        // addresses), since compute_program_hash excludes the tensor volume. Correct by construction —
        // re-derives from the same shared builder create_descriptor() uses, no address inference. See the .cpp.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& c,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    // Structured cache-key tier opt-in (see mesh_device_operation_adapter.hpp). The family packer
    // below is byte-exact by construction -- it reuses the same canonical material the string tier
    // compares -- and its follow-up is the field-by-field encoder that moves the common cases into
    // the 64-byte head with a shared empty tail (zero allocation on the default dispatch).
    static constexpr bool program_cache_uses_structured_key = true;

    static std::optional<program_cache::detail::ProgramCacheStructuredPack> structured_program_cache_pack(
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        const std::vector<ttnn::MeshCoordinate>& coords) {
        namespace pc = program_cache::detail;
        const std::uint16_t family_id =
            pc::StructuredOpRegistry::instance().intern(ttsl::get_type_name<BinaryNgDeviceOperation>());
        if (family_id == 0) {
            return std::nullopt;  // id space exhausted: use the string tier unchanged
        }
        pc::ProgramCacheStructuredPack pack;
        // J271: every byte offset below is READ from the shared head-layout table
        // (pc::kStructuredHeadV2) in program_cache.hpp -- the packer holds no hardcoded offsets.
        // A wire-format change touches that one table plus a version bump, never this code.
        constexpr auto kLayout = pc::kStructuredHeadV2;
        pack.head.store<std::uint16_t>(kLayout.version_offset, pc::kProgramCacheStructuredKeyVersion);
        pack.head.store<std::uint16_t>(kLayout.family_offset, family_id);
        // A single dispatched coordinate fits the head; none or several go to the tail. 0xFF means
        // the tail holds the coordinate set.
        std::vector<std::byte> body;
        if (coords.size() == 1) {
            pack.head.store<std::uint8_t>(kLayout.coord_count_offset, 1);
            pack.head.store<std::uint32_t>(kLayout.coord_x_offset, coords[0][0]);
            pack.head.store<std::uint32_t>(kLayout.coord_y_offset, coords[0][1]);
        } else if (coords.empty()) {
            pack.head.store<std::uint8_t>(kLayout.coord_count_offset, 0);
        } else {
            // Canonicalize the coordinate set: encode in deterministic lexicographic order so two
            // dispatches over the same coordinates in different iteration orders produce identical
            // bytes (J296; a permutation walk over the mesh must not perturb the key). Normally
            // already sorted, so the common case stays allocation-free.
            bool sorted = true;
            for (std::size_t i = 1; i < coords.size(); ++i) {
                if (coords[i] < coords[i - 1]) {
                    sorted = false;
                    break;
                }
            }
            std::vector<ttnn::MeshCoordinate> ordered;
            ttsl::Span<const ttnn::MeshCoordinate> src;
            if (sorted) {
                src = coords;
            } else {
                ordered = coords;
                std::sort(ordered.begin(), ordered.end());
                src = ordered;
            }
            pack.head.store<std::uint8_t>(kLayout.coord_count_offset, 0xFF);
            body.push_back(std::byte{static_cast<std::uint8_t>(coords.size())});
            for (const auto& c : src) {
                const std::uint32_t x = c[0];
                const std::uint32_t y = c[1];
                const auto* xb = reinterpret_cast<const std::byte*>(&x);
                const auto* yb = reinterpret_cast<const std::byte*>(&y);
                body.insert(body.end(), xb, xb + 4);
                body.insert(body.end(), yb, yb + 4);
            }
        }
        // Tail = exact canonical bytes of the material the string tier compares (op identity rides
        // in the head as the interned family id), so (head, tail) is byte-exact by construction.
        // The F154 length prefix keeps the framed block self-describing.
        std::string exact = ttsl::hash::canonical_key(attrs);
        exact += ttsl::hash::canonical_key(tensor_args);
        if (exact.empty() && body.empty()) {
            pack.tail = pc::kEmptyProgramCacheStructuredTail;
        } else {
            auto framed = std::make_shared<std::vector<std::byte>>();
            framed->reserve(2 + exact.size() + body.size());
            const std::uint16_t len = static_cast<std::uint16_t>(exact.size() + body.size());
            framed->push_back(static_cast<std::byte>(len & 0xFF));
            framed->push_back(static_cast<std::byte>((len >> 8) & 0xFF));
            framed->insert(
                framed->end(),
                reinterpret_cast<const std::byte*>(exact.data()),
                reinterpret_cast<const std::byte*>(exact.data() + exact.size()));
            framed->insert(framed->end(), body.begin(), body.end());
            pack.tail = std::move(framed);
        }
        // H232: compose the sub-hash into the u64 discriminator slot at kLayout.sub_offset (the
        // wire format's "tensor hash" field) by reusing the shared H231 fold (program_cache.hpp) --
        // one equation, one definition, shared with the tier's own key fold. Folded in tail-frame
        // byte order (exact, then body = canonicalized coords); a deterministic function of the
        // exact material, so byte-exact on equal dispatches and an internal discriminator
        // independent of the fixed head fields.
        constexpr std::uint64_t kSubSeed = 0xA5A5A5A5A5A5A5A5ULL;
        const std::uint64_t sub = pc::fold_bytes(
            reinterpret_cast<const std::byte*>(body.data()), body.size(),
            pc::fold_bytes(reinterpret_cast<const std::byte*>(exact.data()), exact.size(), kSubSeed));
        pack.head.store<std::uint64_t>(kLayout.sub_offset, sub);
        return pack;
    }

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::prim {

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    float rtol = 0.0f,
    float atol = 0.0f,
    bool equal_nan = false);

ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t binary_ng(
    const Tensor& input_tensor_a_arg,
    ttnn::operations::unary::ScalarVariant scalar,
    ttnn::operations::binary_ng::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    std::optional<ttnn::operations::unary::ScalarVariant> scalar_value = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    bool scalar_is_lhs = false);

}  // namespace ttnn::prim
