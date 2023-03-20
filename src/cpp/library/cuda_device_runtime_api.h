#if !defined(__CUDA_DEVICE_RUNTIME_API_H__)
#define __CUDA_DEVICE_RUNTIME_API_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDABE__)

#if(__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__)
struct cudaFuncAttributes;

__device__ __attribute__((nv_weak)) cudaError_t cudaMalloc(void **p, size_t s)
{
  return cudaErrorUnknown;
}

__device__ __attribute__((nv_weak)) cudaError_t
cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
  return cudaErrorUnknown;
}

__device__ __attribute__((nv_weak)) cudaError_t
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

__device__ __attribute__((nv_weak)) cudaError_t cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

__device__ __attribute__((nv_weak)) cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  int *numBlocks,
  const void *func,
  int blockSize,
  size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

#endif /* (__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__) */

#else /* defined(__CUDABE__) */

#if defined(__cplusplus) &&                                                    \
  defined(__CUDACC__) // Visible to nvcc front-end only
#if !defined(__CUDA_ARCH__) ||                                                 \
  (__CUDA_ARCH__ >= 350) // Visible to SM>=3.5 and "__host__ __device__" only

#include "driver_types.h"
#include "host_defines.h"

extern "C"
{
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaDeviceSynchronize(void);
  /*extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);*/
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaPeekAtLastError(void);

  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaGetDeviceCount(int *count);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaGetDevice(int *device);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaStreamDestroy(cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaEventDestroy(cudaEvent_t event);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaFree(void *devPtr);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaMalloc(void **devPtr, size_t size);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(
    void *dst,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(
    void *devPtr,
    size_t pitch,
    int value,
    size_t width,
    size_t height,
    cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(
    struct cudaPitchedPtr pitchedDevPtr,
    int value,
    struct cudaExtent extent,
    cudaStream_t stream);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaRuntimeGetVersion(int *runtimeVersion);

  extern __device__ __cudart_builtin__ void *CUDARTAPI
  cudaGetParameterBuffer(size_t alignment, size_t size);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice(
    void *func,
    void *parameterBuffer,
    dim3 gridDimension,
    dim3 blockDimension,
    unsigned int sharedMemSize,
    cudaStream_t stream);
  extern __device__ __cudart_builtin__ void *CUDARTAPI cudaGetParameterBufferV2(
    void *func,
    dim3 gridDimension,
    dim3 blockDimension,
    unsigned int sharedMemSize);
  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);

  extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks,
    const void *func,
    int blockSize,
    size_t dynamicSmemSize);
}
namespace
{
template <typename T>
__inline__ __device__ __cudart_builtin__ cudaError_t
cudaMalloc(T **devPtr, size_t size);
template <typename T>
__inline__ __device__ __cudart_builtin__ cudaError_t
cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry);
template <typename T>
__inline__ __device__ __cudart_builtin__ cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  int *numBlocks,
  T func,
  int blockSize,
  size_t dynamicSmemSize);
} // namespace

#endif // !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
#endif // defined(__cplusplus) && defined(__CUDACC__)

#endif /* defined(__CUDABE__) */

#endif /* !__CUDA_DEVICE_RUNTIME_API_H__ */
