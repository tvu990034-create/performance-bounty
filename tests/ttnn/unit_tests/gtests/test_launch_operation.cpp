// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <memory>
#include <type_traits>
#include <vector>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operations/examples/example/device/example_device_operation.hpp"
#include "ttnn/operations/reduction/prod/device/prod_all_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::ttnn::device_operation::mesh_device_operation_utils::all_tensors_have_uniform_storage;
using ::ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates;
using ::ttnn::device_operation::mesh_device_operation_utils::filter_tensor_shards;

// Returns a dummy device tensor with `num_device_shards` populated.
Tensor make_tensor_with_num_shards(int num_device_shards, MeshDevice* mesh_device, int shard_dim = 0) {
    TT_FATAL(num_device_shards > 0 && num_device_shards <= mesh_device->num_devices(), "Invalid number of shards");

    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        ttsl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *distributed::shard_tensor_to_mesh_mapper(*mesh_device, shard_dim),
        *mesh_device);
}

// Returns a dummy device tensor distributed according to the `mapper_config`.
Tensor make_tensor_with_mapper_config(
    int num_device_shards, MeshDevice* mesh_device, const distributed::MeshMapperConfig& mapper_config) {
    auto mapper = distributed::create_mesh_mapper(*mesh_device, mapper_config);
    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        ttsl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *mapper,
        *mesh_device);
}

struct SharedVariables {};
struct OperationAttributes {};

// New-infra style program factory that uses the "create" method (non-heterogeneous dispatch)
struct NewInfraProgramFactory {
    using shared_variables_t = SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_program_t(tt::tt_metal::Program(), SharedVariables{});
    }

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

// New-infra style program factory that uses the "create_at" method (heterogeneous dispatch)
struct NewInfraWorkloadFactory {
    using shared_variables_t = SharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_mesh_workload_t(
            tt::tt_metal::distributed::MeshWorkload(),
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t>());
    }

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<NewInfraWorkloadFactory>);
static_assert(ttnn::device_operation::ProgramFactoryConcept<NewInfraProgramFactory>);

// ---------------------------------------------------------------------------
// ProgramSpecFactoryConcept (Metal 2.0 op-porting stepping stone)
//
// Real ops already use create_program_artifacts (typecast, prod_all, …). The
// checks below still (a) pin the concept's classification (it recognizes a
// create_program_artifacts factory and is mutually exclusive with the other
// factory concepts, per all_factories_valid), and (b) force-instantiate the
// adapter so its bodies are compiled even when a particular TU does not
// instantiate them through a real op.
//
// NOTE: runtime occupancy mapping (programs only on tensor_coords) is covered
// by ProgramSpecAdapterProgramsOnUniformTensorCoords and
// ProgramSpecAdapterProgramsOnlyOnUnevenTensorCoords below. Hit-path TensorArg
// patching / op-owned liveness remain covered by op-level program-cache tests.
// ---------------------------------------------------------------------------
struct ProgramSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const OperationAttributes& /*attrs*/, const Tensor& /*tensor_args*/, Tensor& /*tensor_return_value*/) {
        return ttnn::device_operation::ProgramArtifacts{};
    }
};

// Same, but additionally provides override_runtime_arguments -> classified as the
// custom variant, whose cache-hit path applies the returned ProgramRunArgs via
// UpdateProgramRunArgs.
struct CustomProgramSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const OperationAttributes& /*attrs*/, const Tensor& /*tensor_args*/, Tensor& /*tensor_return_value*/) {
        return ttnn::device_operation::ProgramArtifacts{};
    }
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/ = std::nullopt) {
        return {};
    }
};

// Programs vary across the mesh: one call returns workload-scoped resources plus a program per
// coordinate range.
struct MeshWorkloadSpecFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
        return ttnn::device_operation::MeshWorkloadArtifacts{};
    }
};

// Same, plus a per-range cache-hit run-args refresh.
struct MeshWorkloadSpecFactoryWithOverride {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
        return ttnn::device_operation::MeshWorkloadArtifacts{};
    }
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const OperationAttributes& /*attrs*/,
        const Tensor& /*tensor_args*/,
        Tensor& /*tensor_return_value*/,
        const ttnn::MeshCoordinateRange& /*range*/) {
        return {};
    }
};

// Minimal device operation supplying just the typedefs the adapter inherits.
struct ProgramSpecMinimalOp {
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
};

static_assert(ttnn::device_operation::ProgramSpecFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::MeshWorkloadFactoryConcept<ProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramDescriptorFactoryConcept<ProgramSpecFactory>);

// The custom variant is mutually exclusive with the base one.
static_assert(ttnn::device_operation::CustomProgramSpecFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::MeshWorkloadFactoryConcept<CustomProgramSpecFactory>);
static_assert(!ttnn::device_operation::ProgramDescriptorFactoryConcept<CustomProgramSpecFactory>);

// A mesh-workload factory is its own flavor, and an override doesn't pull it into the others.
static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<MeshWorkloadSpecFactory>);
static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);
static_assert(!ttnn::device_operation::ProgramSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);
static_assert(!ttnn::device_operation::CustomProgramSpecFactoryConcept<MeshWorkloadSpecFactoryWithOverride>);

// Every flavor is exactly one alternative as far as a program_factory_t variant is concerned.
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<ProgramSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<CustomProgramSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<MeshWorkloadSpecFactory>>);
static_assert(ttnn::device_operation::AllFactoriesValid<std::variant<MeshWorkloadSpecFactoryWithOverride>>);

template <typename Factory>
using WorkloadSpecAdapter =
    device_operation::MeshDeviceOperationAdapter<ProgramSpecMinimalOp>::MeshWorkloadSpecFactoryAdapter<Factory>;

// The cache-hit path forks on this; pin both directions.
static_assert(!WorkloadSpecAdapter<MeshWorkloadSpecFactory>::has_override_runtime_arguments());
static_assert(WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::has_override_runtime_arguments());

// Compile-coverage: taking the adapter methods' addresses ODR-uses them, forcing
// the (otherwise un-instantiated) bodies to compile. Never dispatched.
TEST(LaunchOperationTest, ProgramSpecAdapterCompiles) {
    using Adapter = device_operation::MeshDeviceOperationAdapter<
        ProgramSpecMinimalOp>::ProgramSpecMeshWorkloadFactoryAdapter<ProgramSpecFactory>;
    [[maybe_unused]] auto create = &Adapter::create_mesh_workload;
    [[maybe_unused]] auto apply = &Adapter::apply_descriptor;
    [[maybe_unused]] auto resolve = &Adapter::resolve_bindings;

    using CustomAdapter = device_operation::MeshDeviceOperationAdapter<
        ProgramSpecMinimalOp>::CustomProgramSpecMeshWorkloadFactoryAdapter<CustomProgramSpecFactory>;
    [[maybe_unused]] auto ccreate = &CustomAdapter::create_mesh_workload;
    [[maybe_unused]] auto capply = &CustomAdapter::apply_descriptor;

    // The mesh-workload adapter is a separate template; instantiate both its hit-path branches.
    [[maybe_unused]] auto wcreate = &WorkloadSpecAdapter<MeshWorkloadSpecFactory>::create_mesh_workload;
    [[maybe_unused]] auto wapply = &WorkloadSpecAdapter<MeshWorkloadSpecFactory>::apply_descriptor;
    [[maybe_unused]] auto wocreate = &WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::create_mesh_workload;
    [[maybe_unused]] auto woapply = &WorkloadSpecAdapter<MeshWorkloadSpecFactoryWithOverride>::apply_descriptor;
    SUCCEED();
}

// SupportsPerCoreAllocation gates whether launch() will accept a per-core allocated tensor. No op
// declares supports_per_core_allocation today, so the accept path is otherwise never exercised --
// a concept that could never match would look identical at runtime. Pin all three directions here:
// absent, true, and explicitly false.
namespace per_core_opt_in_test {
struct NoDeclaration {};
struct OptedIn {
    static constexpr bool supports_per_core_allocation = true;
};
struct OptedOut {
    static constexpr bool supports_per_core_allocation = false;
};
}  // namespace per_core_opt_in_test

static_assert(!device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::NoDeclaration>);
static_assert(device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::OptedIn>);
static_assert(!device_operation::SupportsPerCoreAllocation<per_core_opt_in_test::OptedOut>);

TEST(LaunchOperationTest, MeshDeviceOperationAdapterGetName) {
    using ::ttnn::operations::examples::ExampleDeviceOperation;
    EXPECT_EQ(
        device_operation::MeshDeviceOperationAdapter<ExampleDeviceOperation>::get_type_name(
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42}),
        "ExampleDeviceOperation");
}

using LaunchOperation2x4Test = tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(LaunchOperation2x4Test, UniformTensor) {
    const tt::tt_metal::TensorSpec tensor_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = ttnn::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));

    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));
}

TEST_F(LaunchOperation2x4Test, UnevenTensor) {
    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());

    EXPECT_THAT(uneven_tensor.device_storage().get_coords(), SizeIs(2));

    EXPECT_FALSE(all_tensors_have_uniform_storage(uneven_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(uneven_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, FilterTensorShards) {
    const tt::tt_metal::TensorSpec tensor_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = ttnn::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first 2 shards and the last 3 shards.
    auto filtered_tensor = filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},
         ttnn::MeshCoordinate{0, 1},
         ttnn::MeshCoordinate{1, 1},
         ttnn::MeshCoordinate{1, 2},
         ttnn::MeshCoordinate{1, 3}},
        full_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(filtered_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first and the last shards.
    filtered_tensor = filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},  //
         ttnn::MeshCoordinate{1, 3}},
        filtered_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(filtered_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{1, 3}));

    // Filter the rest.
    filtered_tensor = filter_tensor_shards(/*tensor_coordinates=*/{}, filtered_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(filtered_tensor));
    EXPECT_THAT(extract_tensor_coordinates(filtered_tensor), IsEmpty());
}

TEST_F(LaunchOperation2x4Test, LaunchOpFilterTensorShards) {
    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_TRUE(all_tensors_have_uniform_storage(sum));
    EXPECT_THAT(
        extract_tensor_coordinates(sum),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(sum_uneven));
    EXPECT_THAT(
        extract_tensor_coordinates(sum_uneven),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, CachingHeterogeneousDispatch) {
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 0);

    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto sum2 = ttnn::add(full_tensor, full_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);

    auto sum3 = ttnn::add(uneven_tensor, uneven_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheGeometryCalculus) {
    namespace pc = tt::tt_metal::program_cache::detail;

    // Knuth bounds for linear probing at alpha = 0.5.
    EXPECT_DOUBLE_EQ(pc::expected_successful_probes(0.5), 1.5);
    EXPECT_DOUBLE_EQ(pc::expected_unsuccessful_probes(0.5), 2.5);

    // A probe budget of 2.0 pins the load factor at 1 - 1/sqrt(3).
    const double alpha = pc::alpha_bound_for_probe_bound(2.0);
    EXPECT_NEAR(alpha, 1.0 - 1.0 / std::sqrt(3.0), 1e-12);

    // Capacity respects the budget and rounds up to a power of two.
    const std::size_t cap = pc::capacity_for_distinct_keys(100, 2.0);
    EXPECT_LE(100.0 / static_cast<double>(cap), alpha);
    EXPECT_EQ(cap & (cap - 1), 0);

    // Cache lines touched by a probe run: offset-aligned slots pack onto lines.
    EXPECT_EQ(pc::lines_touched(0, 0, 64, 64), 0);
    EXPECT_EQ(pc::lines_touched(0, 1, 64, 64), 1);
    EXPECT_EQ(pc::lines_touched(0, 3, 64, 64), 3);
    EXPECT_EQ(pc::lines_touched(1, 2, 64, 128), 2);  // offset 64 + 2*64 bytes -> 2 lines

    // Doubling growth: exact total = (C_max - C_0) * copy, amortized <= c_copy/alpha.
    EXPECT_NEAR(pc::rehash_total(32, 256, 1.0), 224.0, 1e-9);
    EXPECT_LE(pc::rehash_amortized(32, 256, alpha, 1.0), 1.0 / alpha);

    // 10 inserts in 1000 dispatches at equal latency => 1% of dispatch cost.
    EXPECT_NEAR(pc::insertion_fraction(1000, 10, 100.0, 100.0), 0.01, 1e-12);
    EXPECT_NEAR(pc::insert_rate(10, 2.0), 5.0, 1e-12);
    EXPECT_NEAR(pc::miss_rate(1, 1000), 0.001, 1e-12);
    EXPECT_NEAR(pc::total_hot_path_cost(1000, 100.0), 100000.0, 1e-9);

    // Item 26: a sample test is necessary; a disagreement must abort migration.
    const std::vector<std::uint64_t> hash_sample = {5, 42, 100};
    const auto id_hash = [](std::uint64_t x) { return x; };
    const auto id_phi = [](std::uint64_t x) { return x; };
    EXPECT_TRUE(pc::hash_migration_safe(
        hash_sample.begin(), hash_sample.end(), id_hash, id_hash, id_phi));
    const auto perturb_phi = [](std::uint64_t x) { return x ^ 0x1; };
    EXPECT_FALSE(pc::hash_migration_safe(
        hash_sample.begin(), hash_sample.end(), id_hash, id_hash, perturb_phi));
}

TEST_F(LaunchOperation2x4Test, ProgramCacheReserveKeepsMonotonicLayout) {
    auto& program_cache = mesh_device_->get_program_cache();

    // Reserving must not change the observed distinct-key sequence.
    program_cache.reserve(64);
    EXPECT_EQ(program_cache.num_entries(), 0);
    EXPECT_LE(program_cache.expected_insert_probes(), 2.0);

    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    (void)ttnn::add(full_tensor, full_tensor);
    EXPECT_EQ(program_cache.num_entries(), 1);
    (void)ttnn::add(full_tensor, full_tensor);
    EXPECT_EQ(program_cache.num_entries(), 1);

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    (void)ttnn::add(uneven_tensor, uneven_tensor);
    EXPECT_EQ(program_cache.num_entries(), 2);
    (void)ttnn::add(uneven_tensor, uneven_tensor);
    EXPECT_EQ(program_cache.num_entries(), 2);

    // The tag tier stays well below its probe budget after reserve(64): the
    // tier holds 256 slots, so after two inserts the load factor is ~0.008.
    EXPECT_LT(program_cache.tag_load_factor(), 0.5);
    EXPECT_LE(program_cache.expected_insert_probes(), 2.0);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredTierByteExactSemantics) {
    namespace pc = tt::tt_metal::program_cache::detail;

    static_assert(sizeof(pc::ProgramCacheStructuredKey) == pc::ProgramCacheStructuredKey::kSizeBytes);
    static_assert(std::is_standard_layout_v<pc::ProgramCacheStructuredKey>);

    // Typed store/load round-trips byte-exactly.
    pc::ProgramCacheStructuredKey key;
    key.store<std::uint16_t>(0, pc::kProgramCacheStructuredKeyVersion);
    key.store<std::uint16_t>(2, 7);
    key.store<std::uint64_t>(8, 0xDEADBEEFCAFEBABEULL);
    EXPECT_EQ(key.load<std::uint16_t>(0), pc::kProgramCacheStructuredKeyVersion);
    EXPECT_EQ(key.load<std::uint16_t>(2), 7);
    EXPECT_EQ(key.load<std::uint64_t>(8), 0xDEADBEEFCAFEBABEULL);

    // Equality is byte-exact, not semantics-per-field: a single flipped bit changes the key.
    pc::ProgramCacheStructuredKey same = key;
    EXPECT_EQ(key, same);
    EXPECT_FALSE(key != same);
    key.store<std::uint64_t>(8, key.load<std::uint64_t>(8) ^ 0x1);
    EXPECT_NE(key, same);
    EXPECT_FALSE(key == same);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredTierMonotonicAndReserve) {
    namespace pc = tt::tt_metal::program_cache::detail;

    auto make_key = [](std::uint16_t id) {
        pc::ProgramCacheStructuredKey key;
        key.store<std::uint16_t>(0, pc::kProgramCacheStructuredKeyVersion);
        key.store<std::uint16_t>(2, id);
        return key;
    };
    auto make_hash = [](std::uint16_t id) {
        return pc::HashTagSet::avalanche(0x51_ULL ^ static_cast<std::uint64_t>(id));
    };
    auto make_entry = [] {
        tt::tt_metal::Program program;
        struct TrivialSharedVariables {};
        return pc::CachedProgramFactory(
            pc::CachedProgram<TrivialSharedVariables>(std::move(program), TrivialSharedVariables{}), 0);
    };

    pc::ProgramCache program_cache;
    program_cache.reserve(8);
    EXPECT_EQ(program_cache.num_entries(), 0);
    EXPECT_LE(program_cache.structured_load_factor(), 0.5);

    // Distinct keys: monotone 0, 1, 2; duplicates are no-ops and return the resident entry.
    const auto key_a = make_key(1);
    const auto key_b = make_key(2);
    pc::ProgramCacheStructuredTail tail_a =
        std::make_shared<std::vector<std::byte>>(std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}});
    pc::ProgramCacheStructuredTail tail_b =
        std::make_shared<std::vector<std::byte>>(std::vector<std::byte>{std::byte{0x03}});
    auto* first = program_cache.insert_structured(make_hash(1), key_a, tail_a, make_entry());
    EXPECT_EQ(program_cache.num_entries(), 1);
    EXPECT_NE(first, nullptr);
    auto* again = program_cache.insert_structured(make_hash(1), key_a, tail_a, make_entry());
    EXPECT_EQ(program_cache.num_entries(), 1);
    EXPECT_EQ(again, first);
    auto* second = program_cache.insert_structured(make_hash(2), key_b, tail_b, make_entry());
    EXPECT_EQ(program_cache.num_entries(), 2);
    EXPECT_NE(second, first);
    auto* second_again = program_cache.insert_structured(make_hash(2), key_b, tail_b, make_entry());
    EXPECT_EQ(program_cache.num_entries(), 2);
    EXPECT_EQ(second_again, second);

    // The tail refines (never relaxes) the head: same head, different tail is a distinct miss.
    pc::ProgramCacheStructuredTail tail_b_prime =
        std::make_shared<std::vector<std::byte>>(std::vector<std::byte>{std::byte{0x04}});
    EXPECT_EQ(program_cache.num_entries(), 2);
    auto* tail_never_ignored = program_cache.insert_structured(make_hash(2), key_b, tail_b_prime, make_entry());
    EXPECT_EQ(program_cache.num_entries(), 3);
    EXPECT_NE(tail_never_ignored, second);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(2), key_b, tail_b_prime), tail_never_ignored);

    // Exact probe: only the exact head+tail under the exact hash resolves; a neighbor key is a
    // miss, never a wrong hit.
    EXPECT_EQ(program_cache.lookup_structured(make_hash(1), key_a, tail_a), first);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(2), key_b, tail_b), second);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(1), key_a, tail_b), nullptr);

    // Tag gate: the same byte-exact key under a hash that was never inserted is a definite miss.
    EXPECT_EQ(program_cache.lookup_structured(make_hash(9), key_a, tail_a), nullptr);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(9), key_b, tail_b), nullptr);
    EXPECT_FALSE(program_cache.has_hash(make_hash(9)));

    program_cache.clear();
    EXPECT_EQ(program_cache.num_entries(), 0);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(1), key_a, tail_a), nullptr);
    EXPECT_EQ(program_cache.lookup_structured(make_hash(2), key_b, tail_b), nullptr);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredTierGrowKeepsAllEntries) {
    namespace pc = tt::tt_metal::program_cache::detail;

    auto make_key = [](std::uint32_t seed) {
        pc::ProgramCacheStructuredKey key;
        key.store<std::uint16_t>(0, pc::kProgramCacheStructuredKeyVersion);
        key.store<std::uint32_t>(4, seed);
        return key;
    };
    auto make_hash = [](std::uint32_t seed) {
        return pc::HashTagSet::avalanche(0x51_ULL ^ static_cast<std::uint64_t>(seed));
    };
    auto make_entry = [] {
        tt::tt_metal::Program program;
        struct TrivialSharedVariables {};
        return pc::CachedProgramFactory(
            pc::CachedProgram<TrivialSharedVariables>(std::move(program), TrivialSharedVariables{}), 0);
    };

    pc::StructuredProgramCache cache;
    // Push past the initial 64-slot capacity (several doublings) and verify every key still
    // resolves to the exact bytes that were inserted -- rehash must re-probe from the stored
    // placement hashes and never drop or duplicate an entry.
    const std::size_t distinct = pc::StructuredProgramCache::kInitialCapacity + 2;
    for (std::size_t i = 0; i < distinct; ++i) {
        EXPECT_EQ(cache.size(), i);
        EXPECT_NE(cache.insert(make_hash(static_cast<std::uint32_t>(i)), make_key(static_cast<std::uint32_t>(i)),
                               pc::ProgramCacheStructuredTail{}, make_entry()), nullptr);
    }
    EXPECT_EQ(cache.size(), distinct);
    EXPECT_GT(cache.capacity(), distinct);
    // 66 entries in a 128-slot table. The payload tier grows at kPayloadMaxLoadFactor (0.7), so the
    // post-grow load sits in (0.35, 0.7] -- the asset is exact rehash, not a tight upper bound.
    EXPECT_LE(cache.load_factor(), pc::kPayloadMaxLoadFactor);

    for (std::size_t i = 0; i < distinct; ++i) {
        EXPECT_NE(cache.lookup(make_hash(static_cast<std::uint32_t>(i)), make_key(static_cast<std::uint32_t>(i)),
                               pc::ProgramCacheStructuredTail{}), nullptr)
            << "key " << i;
        pc::ProgramCacheStructuredKey wrong = make_key(static_cast<std::uint32_t>(i));
        wrong.store<std::uint8_t>(63, static_cast<std::uint8_t>(i + 1));
        EXPECT_EQ(cache.lookup(make_hash(static_cast<std::uint32_t>(i)), wrong, pc::ProgramCacheStructuredTail{}), nullptr)
            << "bit-flipped key " << i;
        // A hash that was never inserted is a definite gate miss even with a resident key's bytes.
        EXPECT_EQ(cache.lookup(pc::HashTagSet::avalanche(0x9999_ULL ^ static_cast<std::uint64_t>(i)),
                               make_key(static_cast<std::uint32_t>(i)), pc::ProgramCacheStructuredTail{}), nullptr)
            << "wrong-hash gate " << i;
    }
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredOpIdentityInterning) {
    namespace pc = tt::tt_metal::program_cache::detail;
    auto& registry = pc::StructuredOpRegistry::instance();

    const auto id_a = registry.intern("ttnn::prim::BinaryNgDeviceOperation");
    const auto id_b = registry.intern("ttnn::prim::OtherDeviceOperation");
    EXPECT_NE(id_a, 0);
    EXPECT_NE(id_b, 0);
    EXPECT_NE(id_a, id_b);
    // Same name always maps to the same stable id; intern calls carry no per-dispatch string work
    // after the registry's first lookup.
    EXPECT_EQ(registry.intern("ttnn::prim::BinaryNgDeviceOperation"), id_a);
    EXPECT_EQ(registry.intern("ttnn::prim::OtherDeviceOperation"), id_b);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredSeamEmptyTailSharedFastPath) {
    namespace pc = tt::tt_metal::program_cache::detail;

    auto make_key = [](std::uint16_t id) {
        pc::ProgramCacheStructuredKey key;
        key.store<std::uint16_t>(0, pc::kProgramCacheStructuredKeyVersion);
        key.store<std::uint16_t>(2, id);
        return key;
    };
    auto make_hash = [](std::uint16_t id) {
        return pc::HashTagSet::avalanche(0x51_ULL ^ static_cast<std::uint64_t>(id));
    };
    auto make_entry = [] {
        tt::tt_metal::Program program;
        struct TrivialSharedVariables {};
        return pc::CachedProgramFactory(
            pc::CachedProgram<TrivialSharedVariables>(std::move(program), TrivialSharedVariables{}), 0);
    };

    // F155 (omit-all-defaults -> empty tail): the op layer's all-default pack IS the shared
    // singleton, returned by identity, so the common probe compares two shared_ptrs and does zero
    // allocation and zero byte work on the tail term.
    EXPECT_TRUE(pc::structured_key_bytes_equal(
        pc::kEmptyProgramCacheStructuredTail, pc::kEmptyProgramCacheStructuredTail));
    // A store-round-tripped empty tail (equal bytes, distinct block) also resolves: the vector
    // equality branch is the safety path for tails that were serialized, not handed back as-is.
    pc::ProgramCacheStructuredTail store_default = std::make_shared<std::vector<std::byte>>();
    EXPECT_TRUE(pc::structured_key_bytes_equal(store_default, pc::kEmptyProgramCacheStructuredTail));
    // An *unset* tail is not equal to the empty tail: the op layer never probes with an
    // all-default store tail, so "not present" stays distinct from "present but empty".
    EXPECT_FALSE(pc::structured_key_bytes_equal(pc::ProgramCacheStructuredTail{}, pc::kEmptyProgramCacheStructuredTail));

    pc::ProgramCache cache;
    cache.reserve(8);
    const auto key = make_key(1);
    auto* p = cache.insert_structured(
        make_hash(1), key, pc::kEmptyProgramCacheStructuredTail, make_entry());
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(cache.num_entries(), 1);

    // The structured insert registers the SAME tag gate the string tier uses. A tag hit only
    // narrows the candidate set; it never substitutes for the exact (head, tail) probe.
    EXPECT_TRUE(cache.has_hash(make_hash(1)));
    EXPECT_EQ(cache.lookup_structured(make_hash(1), key, pc::kEmptyProgramCacheStructuredTail), p);
    // The op layer may rebuild the tail block per dispatch; the short-circuit is on the singleton.
    EXPECT_EQ(cache.lookup_structured(make_hash(1), key, pc::kEmptyProgramCacheStructuredTail), p);
    EXPECT_EQ(cache.lookup_structured(make_hash(1), key, store_default), p);

    // A hash that was never inserted is a definite miss for the structured path too, regardless of
    // how exact the (head, tail) bytes are.
    EXPECT_FALSE(cache.has_hash(make_hash(9)));
    EXPECT_EQ(cache.lookup_structured(make_hash(9), key, pc::kEmptyProgramCacheStructuredTail), nullptr);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredBirthdayBoundFormula) {
    namespace pc = tt::tt_metal::program_cache::detail;

    // C(n, 2) / space, checked against exact rationals for small spaces.
    EXPECT_NEAR(pc::expected_collision_count(10, 1000), 45.0 / 1000.0, 1e-12);
    EXPECT_NEAR(pc::expected_collision_count(100, 100000), 4950.0 / 100000.0, 1e-12);
    // The 64-bit tag: n(n - 1) / 2^65, i.e. ~2.7e-8 at n = 1M -- the G201 soundness assumption that
    // a tag collision degrades to a rebuild miss at a rate far below one for any realistic n.
    EXPECT_NEAR(pc::expected_collision_count(1000000, ~std::uint64_t{0}), 2.7105e-8, 1e-10);

    // The payload reserve shares the growth load factor: capacity is the pow2-rounded minimal size
    // that fits nu_max keys at alpha = kPayloadMaxLoadFactor.
    EXPECT_EQ(pc::capacity_for_distinct_keys_at_load(100, 0.7), 256u);    // ceil(100/0.7)=143 -> 256
    EXPECT_EQ(pc::capacity_for_distinct_keys_at_load(1000, 0.7), 2048u);  // ceil(1000/0.7)=1429 -> 2048
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredSameHashFloodSpreadsAndResolves) {
    namespace pc = tt::tt_metal::program_cache::detail;

    auto make_key = [](std::uint32_t seed) {
        pc::ProgramCacheStructuredKey key;
        key.store<std::uint16_t>(0, pc::kProgramCacheStructuredKeyVersion);
        key.store<std::uint32_t>(4, seed);
        return key;
    };
    auto make_entry = [] {
        tt::tt_metal::Program program;
        struct TrivialSharedVariables {};
        return pc::CachedProgramFactory(
            pc::CachedProgram<TrivialSharedVariables>(std::move(program), TrivialSharedVariables{}), 0);
    };

    // G205 + H231: a flood of DISTINCT keys that share ONE dispatch hash must still spread across
    // the payload table (placement folds the key bytes, not just the hash) and resolve exactly.
    // The tag tier is dedup-by-value, so 512 distinct keys under a single hash need exactly one tag.
    pc::StructuredProgramCache cache;
    const std::uint64_t shared_hash = pc::HashTagSet::avalanche(0xBEEFULL);
    const std::size_t flood = 512;
    for (std::size_t i = 0; i < flood; ++i) {
        EXPECT_NE(cache.insert(shared_hash, make_key(static_cast<std::uint32_t>(i)),
                               pc::ProgramCacheStructuredTail{}, make_entry()), nullptr)
            << "insert " << i;
    }
    EXPECT_EQ(cache.size(), flood);
    for (std::size_t i = 0; i < flood; ++i) {
        EXPECT_NE(cache.lookup(shared_hash, make_key(static_cast<std::uint32_t>(i)),
                               pc::ProgramCacheStructuredTail{}), nullptr)
            << "lookup " << i;
    }
    // Under the old hash-only placement a same-hash flood would cluster into one long linear run;
    // key-byte placement bounds it. (Probe-depth economics are measured in the host audit, Part H.)
    pc::ProgramCacheStructuredKey wrong = make_key(0);
    wrong.store<std::uint8_t>(63, 0x5A);
    EXPECT_EQ(cache.lookup(shared_hash, wrong, pc::ProgramCacheStructuredTail{}), nullptr);
}

TEST_F(LaunchOperation2x4Test, ProgramCacheStructuredAdapterOptIn) {
    // The mesh adapter's structured branch is gated on TWO compile-time markers on the op family:
    // a flag AND a packer (the flag alone is ignored). The adapter itself always exposes
    // `program_cache_uses_structured_key`, so the launch path's `if constexpr` never mis-reads.
    using BinaryNg = ttnn::operations::binary_ng::BinaryNgDeviceOperation;
    using BinaryNgAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<BinaryNg>;

    static_assert(ttnn::device_operation::has_structured_cache_flag<BinaryNg>::value);
    static_assert(ttnn::device_operation::has_structured_program_cache_pack<BinaryNg>::value);
    static_assert(ttnn::device_operation::structured_cache_key_enabled<BinaryNg>::value);
    static_assert(BinaryNgAdapter::program_cache_uses_structured_key);

    // A legacy op lacking both markers falls back to the string tier without ever being packed.
    using ProdAll = ttnn::prim::ProdAllDeviceOperation;
    using ProdAllAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<ProdAll>;
    static_assert(!ttnn::device_operation::has_structured_cache_flag<ProdAll>::value);
    static_assert(!ttnn::device_operation::has_structured_program_cache_pack<ProdAll>::value);
    static_assert(!ttnn::device_operation::structured_cache_key_enabled<ProdAll>::value);
    static_assert(!ProdAllAdapter::program_cache_uses_structured_key);
}

// Emits a program only for non-idle coordinates, so one create_mesh_workload covers the range split
// and the sharing of workload-scoped resources.
struct PerCoordProdAllFactory {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    static bool is_idle(const ttnn::MeshCoordinate& coord) { return coord[1] % 2 == 1; }

    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const Op::operation_attributes_t& attrs,
        const Op::tensor_args_t& tensor_args,
        Op::tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords) {
        ttnn::device_operation::MeshWorkloadArtifacts artifacts;
        for (const auto& range : tensor_coords.ranges()) {
            for (const auto& coord : range) {
                if (is_idle(coord)) {
                    continue;
                }
                auto program =
                    Op::ProdAllProgramFactory::create_program_artifacts(attrs, tensor_args, tensor_return_value);
                artifacts.programs.push_back(
                    {.range = ttnn::MeshCoordinateRange(coord),
                     .spec = std::move(program.spec),
                     .run_params = std::move(program.run_params)});
            }
        }
        return artifacts;
    }
};

TEST_F(LaunchOperation2x4Test, MeshWorkloadSpecAdapterMapsProgramsPerCoordinate) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter =
        device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<PerCoordProdAllFactory>;
    static_assert(ttnn::device_operation::MeshWorkloadSpecFactoryConcept<PerCoordProdAllFactory>);

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    // One single-coordinate program per non-idle coordinate; idle ones contribute nothing.
    size_t expected = 0;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device_->shape())) {
        const ttnn::MeshCoordinateRange range(coord);
        const bool idle = PerCoordProdAllFactory::is_idle(coord);
        EXPECT_EQ(cached.workload.get_programs().contains(range), !idle) << "coord " << coord;
        EXPECT_EQ(cached.shared_variables.contains(range), !idle) << "coord " << coord;
        expected += idle ? 0 : 1;
    }
    EXPECT_GT(expected, 0u);
    EXPECT_EQ(cached.workload.get_programs().size(), expected);
    EXPECT_EQ(cached.shared_variables.size(), expected);
}

// Same per-coordinate mapping, but with an override so the cache-hit path takes the
// UpdateProgramRunArgs branch instead of the tensor-only refresh.
struct PerCoordProdAllFactoryWithOverride : PerCoordProdAllFactory {
    static std::atomic<int> override_calls;

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const Op::operation_attributes_t& attrs,
        const Op::tensor_args_t& tensor_args,
        Op::tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRange& /*range*/) {
        ++override_calls;
        return Op::ProdAllProgramFactory::create_program_artifacts(attrs, tensor_args, tensor_return_value).run_params;
    }
};
std::atomic<int> PerCoordProdAllFactoryWithOverride::override_calls{0};

TEST_F(LaunchOperation2x4Test, MeshWorkloadSpecAdapterAppliesDescriptorOnCacheHit) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter =
        device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<PerCoordProdAllFactory>;
    using OverrideAdapter = device_operation::MeshDeviceOperationAdapter<Op>::MeshWorkloadSpecFactoryAdapter<
        PerCoordProdAllFactoryWithOverride>;
    static_assert(!Adapter::has_override_runtime_arguments());
    static_assert(OverrideAdapter::has_override_runtime_arguments());

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    // Tensor-refresh branch: a second apply must not disturb the per-coordinate mapping.
    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);
    const auto ranges_before = cached.workload.get_programs().size();
    Adapter::apply_descriptor(cached, attrs, tensor_args, output);
    EXPECT_EQ(cached.workload.get_programs().size(), ranges_before);
    for (const auto& [range, program] : cached.workload.get_programs()) {
        EXPECT_TRUE(cached.shared_variables.contains(range)) << "range " << range;
    }

    // Override branch: called once per range, not once for the whole workload.
    PerCoordProdAllFactoryWithOverride::override_calls = 0;
    auto override_cached = OverrideAdapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);
    const auto override_ranges = override_cached.workload.get_programs().size();
    EXPECT_GT(override_ranges, 1u);
    OverrideAdapter::apply_descriptor(override_cached, attrs, tensor_args, output);
    EXPECT_EQ(PerCoordProdAllFactoryWithOverride::override_calls.load(), static_cast<int>(override_ranges));
}

TEST_F(LaunchOperation2x4Test, ProgramSpecAdapterProgramsOnUniformTensorCoords) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter = device_operation::MeshDeviceOperationAdapter<Op>::ProgramSpecMeshWorkloadFactoryAdapter<
        Op::ProdAllProgramFactory>;

    auto input = make_tensor_with_num_shards(8, mesh_device_.get());
    EXPECT_TRUE(all_tensors_have_uniform_storage(input));

    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    // Fast path in launch: one range covering the entire mesh.
    ttnn::MeshCoordinateRangeSet tensor_coords;
    tensor_coords.merge(ttnn::MeshCoordinateRange(mesh_device_->shape()));

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    const ttnn::MeshCoordinateRange full_mesh(mesh_device_->shape());
    EXPECT_EQ(cached.workload.get_programs().size(), 1u);
    EXPECT_EQ(cached.shared_variables.size(), 1u);
    EXPECT_TRUE(cached.workload.get_programs().contains(full_mesh));
    EXPECT_TRUE(cached.shared_variables.contains(full_mesh));
}

TEST_F(LaunchOperation2x4Test, ProgramSpecAdapterProgramsOnlyOnUnevenTensorCoords) {
    using Op = ttnn::prim::ProdAllDeviceOperation;
    using Adapter = device_operation::MeshDeviceOperationAdapter<Op>::ProgramSpecMeshWorkloadFactoryAdapter<
        Op::ProdAllProgramFactory>;

    auto input = make_tensor_with_num_shards(2, mesh_device_.get());
    EXPECT_FALSE(all_tensors_have_uniform_storage(input));
    EXPECT_THAT(
        extract_tensor_coordinates(input),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));

    Op::operation_attributes_t attrs{.output_mem_config = MemoryConfig{}};
    Op::tensor_args_t tensor_args{.input = input};
    auto output = Op::create_output_tensors(attrs, tensor_args);

    // Slow path in launch: merge occupied coordinates one by one.
    ttnn::MeshCoordinateRangeSet tensor_coords;
    for (const auto& coord : extract_tensor_coordinates(input, mesh_device_.get())) {
        tensor_coords.merge(ttnn::MeshCoordinateRange(coord, coord));
    }

    auto cached = Adapter::create_mesh_workload(attrs, tensor_coords, tensor_args, output);

    EXPECT_EQ(cached.workload.get_programs().size(), tensor_coords.ranges().size());
    EXPECT_EQ(cached.shared_variables.size(), tensor_coords.ranges().size());
    for (const auto& range : tensor_coords.ranges()) {
        EXPECT_TRUE(cached.workload.get_programs().contains(range));
        EXPECT_TRUE(cached.shared_variables.contains(range));
    }

    // Unpopulated mesh coordinates must not have a program associated with them.
    for (const auto& coord : {
             ttnn::MeshCoordinate{0, 2},
             ttnn::MeshCoordinate{0, 3},
             ttnn::MeshCoordinate{1, 0},
             ttnn::MeshCoordinate{1, 1},
             ttnn::MeshCoordinate{1, 2},
             ttnn::MeshCoordinate{1, 3},
         }) {
        for (const auto& [program_range, _] : cached.workload.get_programs()) {
            EXPECT_FALSE(program_range.contains(coord))
                << "unexpected program covering unpopulated mesh coordinate " << coord;
        }
    }
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopology) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get());

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyAugmentedDistribution) {
    auto config_1 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(2, 2),
    };
    auto input_tensor_1 = make_tensor_with_mapper_config(4, mesh_device_.get(), config_1);
    auto config_2 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(1, 4),
    };
    auto input_tensor_2 = make_tensor_with_mapper_config(8, mesh_device_.get(), config_2);
    auto config_3 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(8),
    };
    auto input_tensor_3 = make_tensor_with_mapper_config(16, mesh_device_.get(), config_3);

    auto sum_1 = ttnn::add(input_tensor_1, input_tensor_2);
    auto sum_2 = ttnn::add(input_tensor_2, input_tensor_1);
    auto sum_3 = ttnn::add(input_tensor_3, input_tensor_2);

    EXPECT_EQ(sum_1.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_1.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}}));
    EXPECT_EQ(sum_2.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_2.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
    EXPECT_EQ(sum_3.tensor_topology().distribution_shape(), MeshShape(1, 4));
    EXPECT_EQ(
        sum_3.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyMultipleShardDims) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get(), /*shard_dim=*/1);

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (ttsl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

}  // namespace
}  // namespace ttnn
