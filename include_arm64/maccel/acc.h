// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_ACC_H_
#define MACCEL_ACC_H_

#ifndef _MSC_VER
#include <experimental/propagate_const>
#endif
#include <memory>
#include <string>
#include <vector>

#include "maccel/export.h"
#include "maccel/model.h"

namespace mobilint {
class AcceleratorImpl;

MACCEL_EXPORT const std::string& getMaccelVersion();
MACCEL_EXPORT const std::string& getMaccelGitVersion();
MACCEL_EXPORT const std::string& getMaccelVendor();
MACCEL_EXPORT const std::string& getMaccelProduct();

class MACCEL_EXPORT Accelerator {
public:
    static std::unique_ptr<Accelerator> create(StatusCode& sc);
    static std::unique_ptr<Accelerator> create(int dev_no, StatusCode& sc);
    static std::unique_ptr<Accelerator> create(int dev_no, bool verbose, StatusCode& sc);
    Accelerator(const Accelerator& other) = delete;
    Accelerator(Accelerator&& other) noexcept;
    Accelerator& operator=(const Accelerator& rhs) = delete;
    Accelerator& operator=(Accelerator&& rhs) noexcept;

    /**
     * Destruct the class.
     */
    ~Accelerator();

    /**
     * @brief Retrieve all available cores in NPU.
     *
     * @return Vector list of available cores
     */
    std::vector<Core> getCoreList() const;

    /**
     * @brief Retrieve cores in the specified status.
     *
     * @return Vector list of cores in the specific status
     */
    std::vector<Core> getCoreList(CoreStatus status) const;

    /**
     * @brief Retrieve the current status of the core.
     *
     * @return CoreStatus
     */
    CoreStatus getCoreStatus(Core core) const;

    /**
     * @brief Start event tracing.
     * @deprecated Use mobilint::startTracingEvents(const char*) instead.
     * @return Success/Failure
     */
    bool startTracingEvents(const char* path);

    /**
     * @brief Stop event tracing.
     * @deprecated Use mobilint::stopTracingEvents() instead.
     */
    void stopTracingEvents();

private:
    Accelerator();

#ifndef _MSC_VER
    std::experimental::propagate_const<std::unique_ptr<AcceleratorImpl>> mImpl;
#else
    std::unique_ptr<AcceleratorImpl> mImpl;
#endif

    friend class Model;
};
}  // namespace mobilint

#endif
