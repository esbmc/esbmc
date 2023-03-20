#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 * Function cudaGet_threadIdx()
 *
 * Calculates the thread coordinates from the pthread identifier
 *
 * It assumes a conversion from linear to a tridimensional matrix
 * (as pthread_t is an unsigned long int)
 * */
uint3 cudaGet_threadIdx()
{
  unsigned int linear_value_total = __ESBMC_get_thread_id() - 1;
  unsigned int block_size = blockDim.x * blockDim.y * blockDim.z;
  unsigned int grid_position = (unsigned int)blockIdx.x +
                               blockIdx.y * gridDim.x +
                               blockIdx.z * gridDim.x * gridDim.y;
  unsigned int linear_value = linear_value_total - grid_position * block_size;

  uint3 thread_index;

  thread_index.z = (unsigned int)(linear_value / (blockDim.x * blockDim.y));
  thread_index.y =
    (unsigned int)((linear_value % (blockDim.x * blockDim.y)) / blockDim.x);
  thread_index.x =
    (unsigned int)((linear_value % (blockDim.x * blockDim.y)) % blockDim.x);

  return thread_index;
}

uint3 cudaGet_blockIdx()
{
  unsigned int linear_value = (unsigned int)(__ESBMC_get_thread_id() - 1) /
                              (blockDim.x * blockDim.y * blockDim.z);

  uint3 block_index;
  block_index.z = (unsigned int)(linear_value / (gridDim.x * gridDim.y));
  block_index.y =
    (unsigned int)((linear_value % (gridDim.x * gridDim.y)) / gridDim.x);
  block_index.x =
    (unsigned int)((linear_value % (gridDim.x * gridDim.y)) % gridDim.x);
  return block_index;
}
