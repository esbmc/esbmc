#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__

#include "host_defines.h"

#ifdef __cplusplus

extern "C"
{
#endif

  /**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

  /*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#include <limits.h>
#include <stddef.h>

  enum __device_builtin__ cudaMemcpyKind
  {
    cudaMemcpyHostToHost = 0,     /**< Host   -> Host */
    cudaMemcpyHostToDevice = 1,   /**< Host   -> Device */
    cudaMemcpyDeviceToHost = 2,   /**< Device -> Host */
    cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
    cudaMemcpyDefault = 4 /**< Default based unified virtual address space */
  };

  /**
 * CUDA stream
 */
  /*DEVICE_BUILTIN*/
  typedef struct CUstream_st *cudaStream_t;

  typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

  struct __device_builtin__ cudaDeviceProp
  {
    char name[256];           /**< ASCII string identifying device */
    size_t totalGlobalMem;    /**< Global memory available on device in bytes */
    size_t sharedMemPerBlock; /**< Shared memory available per block in bytes */
    int regsPerBlock;         /**< 32-bit registers available per block */
    int warpSize;             /**< Warp size in threads */
    size_t memPitch; /**< Maximum pitch in bytes allowed by memory copies */
    int maxThreadsPerBlock; /**< Maximum number of threads per block */
    int maxThreadsDim[3];   /**< Maximum size of each dimension of a block */
    int maxGridSize[3];     /**< Maximum size of each dimension of a grid */
    int clockRate;          /**< Clock frequency in kilohertz */
    size_t totalConstMem;   /**< Constant memory available on device in bytes */
    int major;              /**< Major compute capability */
    int minor;              /**< Minor compute capability */
    size_t textureAlignment; /**< Alignment requirement for textures */
    size_t
      texturePitchAlignment; /**< Pitch alignment requirement for texture references bound to pitched memory */
    int
      deviceOverlap; /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int multiProcessorCount; /**< Number of multiprocessors on device */
    int
      kernelExecTimeoutEnabled; /**< Specified whether there is a run time limit on kernels */
    int integrated; /**< Device is integrated as opposed to discrete */
    int
      canMapHostMemory; /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int computeMode;        /**< Compute mode (See ::cudaComputeMode) */
    int maxTexture1D;       /**< Maximum 1D texture size */
    int maxTexture1DMipmap; /**< Maximum 1D mipmapped texture size */
    int
      maxTexture1DLinear; /**< Maximum size for 1D textures bound to linear memory */
    int maxTexture2D[2];       /**< Maximum 2D texture dimensions */
    int maxTexture2DMipmap[2]; /**< Maximum 2D mipmapped texture dimensions */
    int maxTexture2DLinear
      [3]; /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int maxTexture2DGather
      [2]; /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int maxTexture3D[3];        /**< Maximum 3D texture dimensions */
    int maxTexture3DAlt[3];     /**< Maximum alternate 3D texture dimensions */
    int maxTextureCubemap;      /**< Maximum Cubemap texture dimensions */
    int maxTexture1DLayered[2]; /**< Maximum 1D layered texture dimensions */
    int maxTexture2DLayered[3]; /**< Maximum 2D layered texture dimensions */
    int maxTextureCubemapLayered
      [2];               /**< Maximum Cubemap layered texture dimensions */
    int maxSurface1D;    /**< Maximum 1D surface size */
    int maxSurface2D[2]; /**< Maximum 2D surface dimensions */
    int maxSurface3D[3]; /**< Maximum 3D surface dimensions */
    int maxSurface1DLayered[2]; /**< Maximum 1D layered surface dimensions */
    int maxSurface2DLayered[3]; /**< Maximum 2D layered surface dimensions */
    int maxSurfaceCubemap;      /**< Maximum Cubemap surface dimensions */
    int maxSurfaceCubemapLayered
      [2];                   /**< Maximum Cubemap layered surface dimensions */
    size_t surfaceAlignment; /**< Alignment requirements for surfaces */
    int
      concurrentKernels; /**< Device can possibly execute multiple kernels concurrently */
    int ECCEnabled;  /**< Device has ECC support enabled */
    int pciBusID;    /**< PCI bus ID of the device */
    int pciDeviceID; /**< PCI device ID of the device */
    int pciDomainID; /**< PCI domain ID of the device */
    int
      tccDriver; /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int asyncEngineCount; /**< Number of asynchronous engines */
    int
      unifiedAddressing; /**< Device shares a unified address space with the host */
    int memoryClockRate; /**< Peak memory clock frequency in kilohertz */
    int memoryBusWidth;  /**< Global memory bus width in bits */
    int l2CacheSize;     /**< Size of L2 cache in bytes */
    int
      maxThreadsPerMultiProcessor; /**< Maximum resident threads per multiprocessor */
    int streamPrioritiesSupported; /**< Device supports stream priorities */
    int globalL1CacheSupported;    /**< Device supports caching globals in L1 */
    int localL1CacheSupported;     /**< Device supports caching locals in L1 */
    size_t
      sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int
      regsPerMultiprocessor; /**< 32-bit registers available per multiprocessor */
    int
      managedMemory; /**< Device supports allocating managed memory on this system */
    int isMultiGpuBoard; /**< Device is on a multi-GPU board */
    int
      multiGpuBoardGroupID; /**< Unique identifier for a group of devices on the same multi-GPU board */
  };

  /**
 * CUDA device attributes
 */
  enum __device_builtin__ cudaDeviceAttr
  {
    cudaDevAttrMaxThreadsPerBlock =
      1,                         /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX = 2, /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY = 3, /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ = 4, /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock =
      8, /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory =
      9, /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch =
      11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock =
      12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate = 13,        /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap =
      15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount =
      16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout =
      17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory =
      19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode =
      20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth = 21,  /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth = 22,  /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth = 24,  /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth = 26,  /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth =
      27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight =
      28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers =
      29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels =
      31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled = 32,  /**< Device has ECC support enabled */
    cudaDevAttrPciBusId = 33,    /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver = 35,   /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate =
      36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth =
      37,                        /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor =
      39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing =
      41, /**< Device shares a unified address space with the host */
    cudaDevAttrMaxTexture1DLayeredWidth =
      42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers =
      43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth =
      45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight =
      46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt =
      47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt =
      48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt =
      49,                        /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment =
      51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth =
      52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth =
      53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers =
      54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth = 55,  /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth = 56,  /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth = 58,  /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth = 60,  /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth =
      61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers =
      62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth =
      63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight =
      64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers =
      65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth =
      66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth =
      67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers =
      68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth =
      69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth =
      70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight =
      71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch =
      72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth =
      73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight =
      74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor =
      75, /**< Major compute capability version number */
    cudaDevAttrComputeCapabilityMinor =
      76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth =
      77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported =
      78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported =
      79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported =
      80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor =
      81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor =
      82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory =
      83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID =
      85 /**< Unique identifier for a group of devices on the same multi-GPU board */
  };

#ifdef __cplusplus
}

#endif

#endif

#endif /* !__DRIVER_TYPES_H__ */
