// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_NDARRAY_H_
#define MACCEL_NDARRAY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "maccel/export.h"
#include "maccel/type.h"

namespace mobilint {
namespace internal {

inline int64_t numel(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), INT64_C(1), std::multiplies<>());
}

class MACCEL_EXPORT NDArrayData {
public:
    NDArrayData() = default;
    NDArrayData(int64_t bytesize, StatusCode& sc);
    NDArrayData(void* ptr, int64_t bytesize, bool owner);
    ~NDArrayData();
    NDArrayData(const NDArrayData& other);
    NDArrayData& operator=(const NDArrayData& other);

    void* data() const { return mData; }

private:
    void addRef();
    void releaseRef();

    void* mData = nullptr;
    uint8_t* mRefCount = nullptr;
    std::array<uint32_t, 2> mReserved;
    bool mOwner = false;
};

}  // namespace internal

template <typename T>
class NDArray {
public:
    using value_type = T;

    NDArray() = default;
    NDArray(const std::vector<int64_t>& shape, StatusCode& sc)
        : mSize(internal::numel(shape)), mData(sizeof(T) * mSize, sc), mShape(shape) {}

    NDArray(T* ptr, const std::vector<int64_t>& shape, bool owner = false)
        : mSize(internal::numel(shape)),
          mData(ptr, sizeof(T) * mSize, owner),
          mShape(shape) {}

    T* data() const { return reinterpret_cast<T*>(mData.data()); }
    const std::vector<int64_t>& shape() const { return mShape; }
    std::size_t ndim() const { return mShape.size(); }
    int64_t size() const { return mSize; }

    T* begin() { return data(); }
    T* end() { return data() + mSize; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + mSize; }

    T& operator[](size_t idx) { return data()[idx]; }
    T operator[](size_t idx) const { return data()[idx]; }

private:
    int64_t mSize = 0;
    internal::NDArrayData mData;
    std::vector<int64_t> mShape;

    std::array<uint32_t, 4> mReserved;
};

}  // namespace mobilint

#endif  // MACCEL_NDARRAY_H_
