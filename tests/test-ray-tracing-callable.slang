// This block defines the shared resources and compact result layout used by every callable test.
struct CallableResult
{
    uint input;
    uint output;
    float3 value;
    uint tag;
};

uniform RaytracingAccelerationStructure sceneBVH;
uniform RWStructuredBuffer<CallableResult> resultBuffer;

// This block supplies inert hit and miss shaders for tests that execute callables directly from ray generation.
[raypayload]
struct NopRayPayload
{
    uint value : read(caller) : write(caller, closesthit, miss);
};

[shader("closesthit")]
void closestHitCallableNop(inout NopRayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
}

[shader("miss")]
void missCallableNop(inout NopRayPayload payload)
{
}

// These simple callables isolate selection of callable shader-table records from payload ABI behavior.
struct CallableIndexPayload
{
    uint output;
};

[shader("callable")]
void callableSelectFirst(inout CallableIndexPayload payload)
{
    payload.output = 100;
}

[shader("callable")]
void callableSelectSecond(inout CallableIndexPayload payload)
{
    payload.output = 200;
}

[shader("raygeneration")]
void rayGenCallableRuntimeIndex()
{
    uint callableIndex = DispatchRaysIndex().x;
    CallableIndexPayload payload = {0};
    CallShader(callableIndex, payload);

    CallableResult result = {};
    result.input = callableIndex;
    result.output = payload.output;
    resultBuffer[callableIndex] = result;
}

[shader("raygeneration")]
void rayGenCallableNonzeroRecord()
{
    CallableIndexPayload payload = {0};
    CallShader(1, payload);

    CallableResult result = {};
    result.output = payload.output;
    resultBuffer[0] = result;
}

// This block defines a mixed-layout inout payload for focused caller/callee ABI coverage.
struct MixedCallablePayload
{
    uint input;
    float3 value;
    uint tag;
};

[shader("callable")]
void callableAddInput(inout MixedCallablePayload payload)
{
    payload.value += float3(payload.input, payload.input * 2, payload.input * 3);
    payload.tag += payload.input;
}

[shader("raygeneration")]
void rayGenCallableMixedPayload()
{
    MixedCallablePayload payload;
    payload.input = 10;
    payload.value = float3(1, 2, 3);
    payload.tag = 100;
    CallShader(0, payload);

    // Preserve caller-provided and callable-mutated fields for CPU-side ABI checks.
    CallableResult result = {};
    result.input = payload.input;
    result.value = payload.value;
    result.tag = payload.tag;
    resultBuffer[0] = result;
}

// This block defines a 168-byte payload that must use the OptiX direct-callable pointer ABI.
struct LargeCallablePayload
{
    uint values[40];
    uint input;
    uint output;
};

[shader("callable")]
void callableLargePayload(inout LargeCallablePayload payload)
{
    payload.output = payload.input + payload.values[0] + payload.values[20] + payload.values[39];
    payload.values[39] = payload.output * 2;
}

[shader("raygeneration")]
void rayGenCallableLargePayload()
{
    // Initialize the full payload so the callable must read data beyond the 128-byte boundary.
    LargeCallablePayload payload;
    for (uint i = 0; i < 40; ++i)
        payload.values[i] = i + 1;
    payload.input = 7;
    payload.output = 0;
    CallShader(0, payload);

    // Return values from both sides of the large payload to prove round-trip mutation.
    CallableResult result = {};
    result.input = payload.input;
    result.output = payload.output;
    result.tag = payload.values[39];
    resultBuffer[0] = result;
}

// This block defines the ray payload and callable data used to cover every legal caller stage.
[raypayload]
struct StageRayPayload
{
    uint input : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    uint output : read(caller) : write(caller, closesthit, miss);
    uint tag : read(caller) : write(caller, closesthit, miss);
};

struct StageCallablePayload
{
    uint input;
    uint output;
};

[shader("callable")]
void callableFromClosestHit(inout StageCallablePayload payload)
{
    payload.output = payload.input + 100;
}

[shader("callable")]
void callableFromMiss(inout StageCallablePayload payload)
{
    payload.output = payload.input + 200;
}

[shader("callable")]
void callableNestedInner(inout StageCallablePayload payload)
{
    payload.output *= 3;
}

[shader("callable")]
void callableNestedOuter(inout StageCallablePayload payload)
{
    // Invoke another callable record to verify the callable-to-callable stage path.
    payload.output = payload.input + 1;
    CallShader(3, payload);
    payload.output += 2;
}

[shader("closesthit")]
void closestHitCallsCallable(inout StageRayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    // Invoke callable record zero from the closest-hit stage.
    StageCallablePayload callablePayload;
    callablePayload.input = payload.input;
    callablePayload.output = 0;
    CallShader(0, callablePayload);
    payload.output = callablePayload.output;
    payload.tag = 1;
}

[shader("miss")]
void missCallsCallable(inout StageRayPayload payload)
{
    // Invoke callable record one from the miss stage.
    StageCallablePayload callablePayload;
    callablePayload.input = payload.input;
    callablePayload.output = 0;
    CallShader(1, callablePayload);
    payload.output = callablePayload.output;
    payload.tag = 2;
}

[shader("raygeneration")]
void rayGenCallableStages()
{
    uint dispatchIndex = DispatchRaysIndex().x;
    StageRayPayload payload = {10 * (dispatchIndex + 1), 0, 0};

    // Send the first ray through closest-hit, the second through miss, and the third through nested callables.
    if (dispatchIndex < 2)
    {
        RayDesc ray;
        ray.Origin = dispatchIndex == 0 ? float3(0.25, 0.25, 0.0) : float3(2.0, 2.0, 0.0);
        ray.Direction = float3(0.0, 0.0, 1.0);
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);
    }
    else
    {
        StageCallablePayload callablePayload;
        callablePayload.input = payload.input;
        callablePayload.output = 0;
        CallShader(2, callablePayload);
        payload.output = callablePayload.output;
        payload.tag = 3;
    }

    // Record which stage path ran and the final callable mutation.
    CallableResult result = {};
    result.input = payload.input;
    result.output = payload.output;
    result.tag = payload.tag;
    resultBuffer[dispatchIndex] = result;
}

// This block verifies that a callable uniform parameter is sourced from its shader-table record.
struct ShaderRecordCallablePayload
{
    uint input;
    uint output;
};

[shader("callable")]
void callableShaderRecord(inout ShaderRecordCallablePayload payload, uniform uint recordValue)
{
    payload.output = payload.input + recordValue;
}

[shader("raygeneration")]
void rayGenCallableShaderRecord()
{
    // Invoke the record-data callable with a recognizable input pattern.
    ShaderRecordCallablePayload payload = {0x01020304, 0};
    CallShader(0, payload);

    // Return both the computed output and the inferred record contribution.
    CallableResult result = {};
    result.input = payload.input;
    result.output = payload.output;
    result.tag = payload.output - payload.input;
    resultBuffer[0] = result;
}
