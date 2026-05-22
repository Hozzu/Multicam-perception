// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_TYPE_H_
#define MACCEL_TYPE_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "maccel/export.h"

namespace mobilint {

enum class StatusCode;
inline bool operator!(StatusCode sc) { return static_cast<int>(sc) != 0; }

enum class Cluster : int32_t {
    Cluster0 = 1 << 16,
    Cluster1 = 2 << 16,
    Error = 0x7FFF'0000,
};

enum class Core : int32_t {
    Core0 = 1,
    Core1 = 2,
    Core2 = 3,
    Core3 = 4,
    All = 0x0000'FFFC,
    GlobalCore = 0x0000'FFFE,
    Error = 0x0000'FFFF,
};

enum class CollaborationModel {
    Unified,
    Separated,
    Undefined,
};

enum class CoreStatus {
    Vacant,
    Ready,
    Idle,
    Running,
};

enum class SchedulePolicy {
    FIFO,
    LIFO,
    ByPriority,
    Undefined,
};

enum class LatencySetPolicy {
    Auto,
    Manual,
};

enum class MaintenancePolicy {
    Maintain,
    DropExpired,
    Undefined,
};

enum class InferenceResult {
    Successful,
    Expired,
    Unexpected,
};

struct Scale {
    std::vector<float> scale_list;
    float scale = 0.0F;
    bool is_uniform = false;

    float operator[](int i) const {
        if (is_uniform) {
            return scale;
        }
        return scale_list[i];
    }
};

class Statistics;

struct CoreId {
    Cluster cluster = Cluster::Error;
    Core core = Core::Error;

    bool operator==(const CoreId& rhs) const {
        return std::tie(cluster, core) == std::tie(rhs.cluster, rhs.core);
    }

    bool operator<(const CoreId& rhs) const {
        return std::tie(cluster, core) < std::tie(rhs.cluster, rhs.core);
    }
};

/**
 * @brief Buffer
 *
 */
struct Buffer {
    int8_t* data = nullptr;
    uint64_t size = 0;
};

enum class CoreMode : uint8_t {
    Single = 0,
    Multi = 1,
    Global = 2,
};

struct BufferInfo {
    uint32_t original_height = 0;
    uint32_t original_width = 0;
    uint32_t original_channel = 0;
    uint32_t reshaped_height = 0;
    uint32_t reshaped_width = 0;
    uint32_t reshaped_channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t channel = 0;

    uint32_t original_size() const {
        return original_height * original_width * original_channel;
    }
    uint32_t reshaped_size() const {
        return reshaped_height * reshaped_width * reshaped_channel;
    }
    uint32_t size() const { return height * width * channel; }
};

class MACCEL_EXPORT ModelConfig {
public:
    ModelConfig();

    bool includeAllCores();
    bool excludeAllCores();

    bool include(Cluster cluster, Core core);
    bool include(Cluster cluster);
    bool include(Core core);

    bool exclude(Cluster cluster, Core core);
    bool exclude(Cluster cluster);
    bool exclude(Core core);

    bool setGlobalMode(Cluster cluster);
    bool setMultiMode(Cluster cluster);
    CoreMode getCoreMode() const { return mCoreMode; }

    const std::vector<CoreId>& getCoreIds() const { return mCoreIds; }

    SchedulePolicy schedule_policy = SchedulePolicy::FIFO;
    LatencySetPolicy latency_set_policy = LatencySetPolicy::Auto;
    MaintenancePolicy maintenance_policy = MaintenancePolicy::Maintain;
    std::vector<uint64_t> early_latencies;
    std::vector<uint64_t> finish_latencies;

private:
    std::vector<CoreId> mCoreIds;
    CoreMode mCoreMode = CoreMode::Single;
};

/**
 * @brief LogLevel
 */
enum class LogLevel : char {
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERR = 4,
    FATAL = 5,
    OFF = 6,
};

MACCEL_EXPORT void setLogLevel(LogLevel level);

/**
 * @brief Event tracer APIs.
 */
MACCEL_EXPORT bool startTracingEvents(const char*);
MACCEL_EXPORT void stopTracingEvents();

}  // namespace mobilint

#endif
