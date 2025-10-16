#if !defined(__DEVICE_LAUNCH_PARAMETERS_H__)
#  define __DEVICE_LAUNCH_PARAMETERS_H__

#  include <stddef.h>
#  include <stdlib.h>
#  include "vector_types.h"

#  include <pthread.h>

pthread_t __ESBMC_get_thread_id(void);

#  if !defined(__STORAGE__)

#    define __STORAGE__ extern const

#  endif /* __STORAGE__ */

#  ifdef __cplusplus

extern "C"
{
#  endif

  /*__device_builtin__ __STORAGE__ */ dim3 blockDim;

  /*__device_builtin__ __STORAGE__ */ dim3 gridDim;

  __device_builtin__ __STORAGE__ int warpSize = 32;

  uint3 indexOfThread[1024];
  uint3 indexOfBlock[1024];

  uint3 getThreadIdx(unsigned int id);
  uint3 getBlockIdx(unsigned int id);

  uint3 getThreadIdx_table(unsigned int id);
  uint3 getBlockIdx_table(unsigned int id);

  void assignIndexes()
  {
    __ESBMC_atomic_begin();
    int i;

    //for(i = 0; i < GPU_threads; i++)
    for (i = 0; i < 2; i++)
      indexOfBlock[i] = getBlockIdx_table(i);
    //for(i = 0; i < GPU_threads; i++)
    for (i = 0; i < 2; i++)
      indexOfThread[i] = getThreadIdx_table(i);

    __ESBMC_atomic_end();
  }

#  define threadIdx indexOfThread[__ESBMC_get_thread_id() - 1]
#  define blockIdx indexOfBlock[__ESBMC_get_thread_id() - 1]

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

  /**
   * Apply a lookup table to drastically reduce validation time 
   * while retaining the same behavior
   */
  uint3 getThreadIdx_table(unsigned int id)
  {
    __ESBMC_atomic_begin();

    unsigned int lookup3Dx[2][2][2] = {{{0, 0}, {0, 0}}, {{0, 1}, {0, 1}}};
    unsigned int lookup3Dz[2][2][2] = {{{0, 0}, {0, 0}}, {{1, 0}, {1, 0}}};

    unsigned int gridDim_index = (unsigned int)(gridDim.x - 1);
    unsigned int blockDim_index = (unsigned int)(blockDim.x - 1);

    uint3 thread_index;

    thread_index.z =
      (unsigned int)(lookup3Dz[id][gridDim_index][blockDim_index]);
    thread_index.y = (unsigned int)0;
    thread_index.x =
      (unsigned int)(lookup3Dx[id][gridDim_index][blockDim_index]);

    __ESBMC_atomic_end();
    return thread_index;
  }

  uint3 getBlockIdx_table(unsigned int id)
  {
    __ESBMC_atomic_begin();

    unsigned int lookup3D[2][2][2] = {{{0, 0}, {0, 0}}, {{1, 0}, {1, 0}}};

    unsigned int gridDim_index = (unsigned int)(gridDim.x - 1);
    unsigned int blockDim_index = (unsigned int)(blockDim.x - 1);

    uint3 block_index;

    block_index.z = (unsigned int)0;
    block_index.y = (unsigned int)0;
    block_index.x = (unsigned int)(lookup3D[id][gridDim_index][blockDim_index]);

    __ESBMC_atomic_end();
    return block_index;
  }

#  undef __STORAGE__

#  ifdef __cplusplus
}
#  endif

#endif /* !__DEVICE_LAUNCH_PARAMETERS_H__ */
