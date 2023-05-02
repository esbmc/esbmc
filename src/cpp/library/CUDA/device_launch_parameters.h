#if !defined(__DEVICE_LAUNCH_PARAMETERS_H__)
#define __DEVICE_LAUNCH_PARAMETERS_H__

#include <stddef.h>
#include <stdlib.h>
#include "vector_types.h"

#include <pthread.h>

unsigned int __ESBMC_get_thread_id(void);

#if !defined(__STORAGE__)

#define __STORAGE__ extern const

#endif /* __STORAGE__ */

#ifdef __cplusplus

extern "C"
{
#endif

  /*__device_builtin__ __STORAGE__ */ dim3 blockDim;

  /*__device_builtin__ __STORAGE__ */ dim3 gridDim;

  __device_builtin__ __STORAGE__ int warpSize = 32;

  /*
uint3 __device_builtin__ __STORAGE__ threadIdx;
uint3 __device_builtin__ __STORAGE__ blockIdx;
*/

  uint3 indexOfThread[1024];
  uint3 indexOfBlock[1024];

  uint3 getThreadIdx(unsigned int id);
  uint3 getBlockIdx(unsigned int id);

  void assignIndexes()
  {
    __ESBMC_atomic_begin();
    int i;
    //for(i = 0; i < GPU_threads; i++)
    for(i = 0; i < 2; i++)
      indexOfBlock[i] = getBlockIdx(i);
    //for(i = 0; i < GPU_threads; i++)
    for(i = 0; i < 2; i++)
      indexOfThread[i] = getThreadIdx(i);
    __ESBMC_atomic_end();
  }

#define threadIdx indexOfThread[__ESBMC_get_thread_id() - 1]
#define blockIdx indexOfBlock[__ESBMC_get_thread_id() - 1]

  uint3 getThreadIdx(unsigned int id)
  {
    __ESBMC_atomic_begin();
    unsigned int linear_value_total = id;
    unsigned int block_size = blockDim.x * blockDim.y * blockDim.z;
    unsigned int grid_position = (unsigned int)indexOfBlock[id].x +
                                 indexOfBlock[id].y * gridDim.x +
                                 indexOfBlock[id].z * gridDim.x * gridDim.y;
    unsigned int linear_value = linear_value_total - grid_position * block_size;

    uint3 thread_index;

    thread_index.z = (unsigned int)(linear_value / (blockDim.x * blockDim.y));
    thread_index.y =
      (unsigned int)((linear_value % (blockDim.x * blockDim.y)) / blockDim.x);
    thread_index.x =
      (unsigned int)((linear_value % (blockDim.x * blockDim.y)) % blockDim.x);

    __ESBMC_atomic_end();
    return thread_index;
  }

  uint3 getBlockIdx(unsigned int id)
  {
    __ESBMC_atomic_begin();

    unsigned int linear_value =
      (unsigned int)id / (blockDim.x * blockDim.y * blockDim.z);

    uint3 block_index;
    block_index.z = (unsigned int)(linear_value / (gridDim.x * gridDim.y));
    block_index.y =
      (unsigned int)((linear_value % (gridDim.x * gridDim.y)) / gridDim.x);
    block_index.x =
      (unsigned int)((linear_value % (gridDim.x * gridDim.y)) % gridDim.x);

    __ESBMC_atomic_end();
    return block_index;
  }
  extern "C++"
  {
    dim3 cudaGet_blockDim();
    dim3 cudaGet_gridDim();
  }
  /*
#define blockDim cudaGet_blockDim()
#define gridDim cudaGet_gridDim();
*/

#undef __STORAGE__

#ifdef __cplusplus
}

#endif

#if 0

#if !defined(__cudaGet_threadIdx)

#define __cudaGet_threadIdx() threadIdx

#endif /* __cudaGet_threadIdx */

#if !defined(__cudaGet_blockIdx)

#define __cudaGet_blockIdx() blockIdx

#endif /* __cudaGet_blockIdx */

#if !defined(__cudaGet_blockDim)

#define __cudaGet_blockDim() blockDim

#endif /* __cudaGet_blockDim */

#if !defined(__cudaGet_gridDim)

#define __cudaGet_gridDim() gridDim

#endif /* __cudaGet_gridDim */

#if !defined(__cudaGet_warpSize)

#define __cudaGet_warpSize() warpSize

#endif /* __cudaGet_warpSize */
#endif

#endif /* !__DEVICE_LAUNCH_PARAMETERS_H__ */
