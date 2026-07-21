#include "testing.h"
#include "test-ray-tracing-common.h"

#include <cstring>
#include <vector>

using namespace rhi;
using namespace rhi::testing;

namespace {

// This block mirrors the compact shader result type used by every callable test.
struct CallableResult
{
    uint32_t input;
    uint32_t output;
    float value[3];
    uint32_t tag;
};

static_assert(sizeof(CallableResult) == 24);

// This helper creates a structured output buffer large enough for one result per ray.
ComPtr<IBuffer> createResultBuffer(IDevice* device, uint32_t resultCount)
{
    BufferDesc desc = {};
    desc.size = sizeof(CallableResult) * resultCount;
    desc.elementSize = sizeof(CallableResult);
    desc.memoryType = MemoryType::DeviceLocal;
    desc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;

    ComPtr<IBuffer> resultBuffer = device->createBuffer(desc);
    REQUIRE(resultBuffer != nullptr);
    return resultBuffer;
}

// This helper binds the shared resources and launches the requested number of ray-generation invocations.
void dispatchCallablePipeline(
    ICommandQueue* queue,
    IRayTracingPipeline* pipeline,
    IShaderTable* shaderTable,
    IBuffer* resultBuffer,
    IAccelerationStructure* tlas,
    uint32_t width
)
{
    auto commandEncoder = queue->createCommandEncoder();
    auto passEncoder = commandEncoder->beginRayTracingPass();
    auto rootObject = passEncoder->bindPipeline(pipeline, shaderTable);
    auto cursor = ShaderCursor(rootObject);
    cursor["resultBuffer"].setBinding(resultBuffer);
    cursor["sceneBVH"].setBinding(tlas);
    passEncoder->dispatchRays(0, width, 1, 1);
    passEncoder->end();

    REQUIRE_CALL(queue->submit(commandEncoder->finish()));
    REQUIRE_CALL(queue->waitOnHost());
}

// This helper copies the GPU results into ordinary CPU storage for focused assertions.
std::vector<CallableResult> readResults(IDevice* device, IBuffer* resultBuffer, uint32_t resultCount)
{
    ComPtr<ISlangBlob> resultBlob;
    REQUIRE_CALL(device->readBuffer(resultBuffer, 0, sizeof(CallableResult) * resultCount, resultBlob.writeRef()));

    std::vector<CallableResult> results(resultCount);
    std::memcpy(results.data(), resultBlob->getBufferPointer(), sizeof(CallableResult) * resultCount);
    return results;
}

// This helper checks the vector portion of a result without hiding which component failed.
void checkFloat3(const float* actual, float x, float y, float z)
{
    CHECK_EQ(actual[0], x);
    CHECK_EQ(actual[1], y);
    CHECK_EQ(actual[2], z);
}

} // namespace

// This test selects callable records using a runtime value from the dispatch.
GPU_TEST_CASE("ray-tracing-callable-runtime-index", D3D12 | Vulkan | CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    // Build the common acceleration structures required by the ray-tracing pipeline.
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 2);

    // Build two callable SBT records so the dispatch index selects a distinct result at runtime.
    std::vector<const char*> raygenNames = {"rayGenCallableRuntimeIndex"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallableNop", nullptr}};
    std::vector<const char*> missNames = {"missCallableNop"};
    std::vector<const char*> callableNames = {"callableSelectFirst", "callableSelectSecond"};
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames
    );

    // Dispatch two rays and verify that the runtime index selects the corresponding callable.
    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 2);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 2);

    CHECK_EQ(results[0].input, 0);
    CHECK_EQ(results[0].output, 100);
    CHECK_EQ(results[1].input, 1);
    CHECK_EQ(results[1].output, 200);
}

// This test calls a nonzero callable record directly, independently of runtime indexing.
GPU_TEST_CASE("ray-tracing-callable-nonzero-sbt-record", D3D12 | Vulkan | CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 1);

    // Keep a record at index zero so the tested callable occupies SBT record one.
    std::vector<const char*> raygenNames = {"rayGenCallableNonzeroRecord"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallableNop", nullptr}};
    std::vector<const char*> missNames = {"missCallableNop"};
    std::vector<const char*> callableNames = {"callableSelectFirst", "callableSelectSecond"};
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames
    );

    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 1);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 1);

    CHECK_EQ(results[0].output, 200);
}

// This test round-trips a mixed-layout payload through a callable shader.
GPU_TEST_CASE("ray-tracing-callable-mixed-payload", D3D12 | Vulkan | CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 1);

    std::vector<const char*> raygenNames = {"rayGenCallableMixedPayload"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallableNop", nullptr}};
    std::vector<const char*> missNames = {"missCallableNop"};
    std::vector<const char*> callableNames = {"callableAddInput"};
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames
    );

    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 1);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 1);

    CHECK_EQ(results[0].input, 10);
    checkFloat3(results[0].value, 11.0f, 22.0f, 33.0f);
    CHECK_EQ(results[0].tag, 110);
}

// This CUDA-specific test proves that direct-callable data is not constrained by the 128-byte ray-payload limit.
GPU_TEST_CASE("ray-tracing-callable-large-payload", CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    // Build the minimal pipeline resources needed to execute one large-payload callable.
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 1);

    // Use a dedicated callable entry point whose payload exceeds the ray-payload register budget.
    std::vector<const char*> raygenNames = {"rayGenCallableLargePayload"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallableNop", nullptr}};
    std::vector<const char*> missNames = {"missCallableNop"};
    std::vector<const char*> callableNames = {"callableLargePayload"};
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames
    );

    // Verify reads and writes at both ends of the oversized callable payload.
    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 1);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 1);

    CHECK_EQ(results[0].input, 7);
    CHECK_EQ(results[0].output, 69);
    CHECK_EQ(results[0].tag, 138);
}

// This test executes CallShader from closest-hit, miss, and another callable shader.
GPU_TEST_CASE("ray-tracing-callable-stages", D3D12 | Vulkan | CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    // Build one triangle so adjacent invocations can intentionally take hit and miss paths.
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 3);

    // Place the stage-specific and nested callable shaders at known SBT indices.
    std::vector<const char*> raygenNames = {"rayGenCallableStages"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallsCallable", nullptr}};
    std::vector<const char*> missNames = {"missCallsCallable"};
    std::vector<const char*> callableNames = {
        "callableFromClosestHit",
        "callableFromMiss",
        "callableNestedOuter",
        "callableNestedInner",
    };
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames
    );

    // Dispatch a hit, a miss, and a nested callable invocation, then identify each completed path.
    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 3);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 3);

    CHECK_EQ(results[0].input, 10);
    CHECK_EQ(results[0].output, 110);
    CHECK_EQ(results[0].tag, 1);

    CHECK_EQ(results[1].input, 20);
    CHECK_EQ(results[1].output, 220);
    CHECK_EQ(results[1].tag, 2);

    CHECK_EQ(results[2].input, 30);
    CHECK_EQ(results[2].output, 95);
    CHECK_EQ(results[2].tag, 3);
}

// This CUDA test verifies that callable shader-record data is copied after the OptiX 32-byte record header.
GPU_TEST_CASE("ray-tracing-callable-shader-record", CUDA)
{
    if (!device->hasFeature(Feature::RayTracing))
        SKIP("ray tracing not supported");

    // Build the minimal resources needed to launch the record-data callable.
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);
    SingleTriangleBLAS blas(device, queue, false);
    TLAS tlas(device, queue, blas.blas);
    ComPtr<IBuffer> resultBuffer = createResultBuffer(device, 1);

    // Populate the callable record immediately after the OptiX shader identifier header.
    const uint32_t recordValue = 0x12345678;
    ShaderRecordOverwrite callableRecord = {};
    callableRecord.offset = 32;
    callableRecord.size = sizeof(recordValue);
    std::memcpy(callableRecord.data, &recordValue, sizeof(recordValue));

    // Build a pipeline whose callable reads its uniform parameter directly from the shader record.
    std::vector<const char*> raygenNames = {"rayGenCallableShaderRecord"};
    std::vector<HitGroupProgramNames> hitGroupNames = {{"closestHitCallableNop", nullptr}};
    std::vector<const char*> missNames = {"missCallableNop"};
    std::vector<const char*> callableNames = {"callableShaderRecord"};
    RayTracingTestPipeline pipeline(
        device,
        "test-ray-tracing-callable",
        raygenNames,
        hitGroupNames,
        missNames,
        RayTracingPipelineFlags::None,
        nullptr,
        callableNames,
        &callableRecord
    );

    // Verify that the callable combines caller data with the value stored in its SBT record.
    dispatchCallablePipeline(queue, pipeline.raytracingPipeline, pipeline.shaderTable, resultBuffer, tlas.tlas, 1);
    const std::vector<CallableResult> results = readResults(device, resultBuffer, 1);

    CHECK_EQ(results[0].input, 0x01020304);
    CHECK_EQ(results[0].output, 0x1336597c);
    CHECK_EQ(results[0].tag, recordValue);
}
