#ifndef _CUDA_RUNTIME_API_H
#define _CUDA_RUNTIME_API_H 1

#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <driver_types.h>

#include "host_defines.h"
#include "builtin_types.h"
#include "cuda_device_runtime_api.h"

/** \cond impl_private */
#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v)

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

void *address[10];
int counter = 0;

void verifCopyKind(void *dst, const void *src, enum cudaMemcpyKind kind)
{
  int typeTransf = -1;

  for(int i = 0; i < 10; i++)
  {
    if(dst == address[i])
    {
      typeTransf = 1;
    }
    else if(src == address[i])
    {
      typeTransf = 2;
    }
  }
  __ESBMC_assert(typeTransf == kind, "Direction of the copy incorrect");
}

/**
 * Error codes
 */
typedef enum cudaError
{
  /**
	 * The API call returned with no errors. In the case of query calls, this
	 * can also mean that the operation being queried is complete (see
	 * ::cuEventQuery() and ::cuStreamQuery()).
	 */
  CUDA_SUCCESS = 0,
  cudaSuccess = 0,

  /**
	 * This indicates that one or more of the parameters passed to the API call
	 * is not within an acceptable range of values.
	 */
  CUDA_ERROR_INVALID_VALUE = 1,

  /**
	 * The API call failed because it was unable to allocate enough memory to
	 * perform the requested operation.
	 */
  CUDA_ERROR_OUT_OF_MEMORY = 2,

  /**
	 * This indicates that the CUDA driver has not been initialized with
	 * ::cuInit() or that initialization has failed.
	 */
  CUDA_ERROR_NOT_INITIALIZED = 3,

  /**
	 * This indicates that the CUDA driver is in the process of shutting down.
	 */
  CUDA_ERROR_DEINITIALIZED = 4,

  /**
	 * This indicates profiler is not initialized for this run. This can
	 * happen when the application is running with external profiling tools
	 * like visual profiler.
	 */
  CUDA_ERROR_PROFILER_DISABLED = 5,

  /**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to attempt to enable/disable the profiling via ::cuProfilerStart or
	 * ::cuProfilerStop without initialization.
	 */
  CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,

  /**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call cuProfilerStart() when profiling is already enabled.
	 */
  CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,

  /**
	 * \deprecated
	 * This error return is deprecated as of CUDA 5.0. It is no longer an error
	 * to call cuProfilerStop() when profiling is already disabled.
	 */
  CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,

  /**
	 * This indicates that the device ordinal supplied by the user does not
	 * correspond to a valid CUDA device.
	 */
  cudaErrorInvalidDevice = 10,

  /*
	 * This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
	 */
  cudaErrorDeviceAlreadyInUse = 54,

  /**
	 * This indicates that no CUDA-capable devices were detected by the installed
	 * CUDA driver.
	 */
  CUDA_ERROR_NO_DEVICE = 100,

  /**
	 * This indicates that the device ordinal supplied by the user does not
	 * correspond to a valid CUDA device.
	 */
  CUDA_ERROR_INVALID_DEVICE = 101,

  /**
	 * This indicates that the device kernel image is invalid. This can also
	 * indicate an invalid CUDA module.
	 */
  CUDA_ERROR_INVALID_IMAGE = 200,

  /**
	 * This most frequently indicates that there is no context bound to the
	 * current thread. This can also be returned if the context passed to an
	 * API call is not a valid handle (such as a context that has had
	 * ::cuCtxDestroy() invoked on it). This can also be returned if a user
	 * mixes different API versions (i.e. 3010 context with 3020 API calls).
	 * See ::cuCtxGetApiVersion() for more details.
	 */
  CUDA_ERROR_INVALID_CONTEXT = 201,

  /**
	 * This indicated that the context being supplied as a parameter to the
	 * API call was already the active context.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.2. It is no longer an
	 * error to attempt to push the active context via ::cuCtxPushCurrent().
	 */
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,

  /**
	 * This indicates that a map or register operation has failed.
	 */
  CUDA_ERROR_MAP_FAILED = 205,

  /**
	 * This indicates that an unmap or unregister operation has failed.
	 */
  CUDA_ERROR_UNMAP_FAILED = 206,

  /**
	 * This indicates that the specified array is currently mapped and thus
	 * cannot be destroyed.
	 */
  CUDA_ERROR_ARRAY_IS_MAPPED = 207,

  /**
	 * This indicates that the resource is already mapped.
	 */
  CUDA_ERROR_ALREADY_MAPPED = 208,

  /**
	 * This indicates that there is no kernel image available that is suitable
	 * for the device. This can occur when a user specifies code generation
	 * options for a particular CUDA source file that do not include the
	 * corresponding device configuration.
	 */
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,

  /**
	 * This indicates that a resource has already been acquired.
	 */
  CUDA_ERROR_ALREADY_ACQUIRED = 210,

  /**
	 * This indicates that a resource is not mapped.
	 */
  CUDA_ERROR_NOT_MAPPED = 211,

  /**
	 * This indicates that a mapped resource is not available for access as an
	 * array.
	 */
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,

  /**
	 * This indicates that a mapped resource is not available for access as a
	 * pointer.
	 */
  CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,

  /**
	 * This indicates that an uncorrectable ECC error was detected during
	 * execution.
	 */
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,

  /**
	 * This indicates that the ::CUlimit passed to the API call is not
	 * supported by the active device.
	 */
  CUDA_ERROR_UNSUPPORTED_LIMIT = 215,

  /**
	 * This indicates that the ::CUcontext passed to the API call can
	 * only be bound to a single CPU thread at a time but is already
	 * bound to a CPU thread.
	 */
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,

  /**
	 * This indicates that peer access is not supported across the given
	 * devices.
	 */
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,

  /**
	 * This indicates that a PTX JIT compilation failed.
	 */
  CUDA_ERROR_INVALID_PTX = 218,

  /**
	 * This indicates an error with OpenGL or DirectX context.
	 */
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,

  /**
	 * This indicates that the device kernel source is invalid.
	 */
  CUDA_ERROR_INVALID_SOURCE = 300,

  /**
	 * This indicates that the file specified was not found.
	 */
  CUDA_ERROR_FILE_NOT_FOUND = 301,

  /**
	 * This indicates that a link to a shared object failed to resolve.
	 */
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

  /**
	 * This indicates that initialization of a shared object failed.
	 */
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,

  /**
	 * This indicates that an OS call failed.
	 */
  CUDA_ERROR_OPERATING_SYSTEM = 304,

  /**
	 * This indicates that a resource handle passed to the API call was not
	 * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
	 */
  CUDA_ERROR_INVALID_HANDLE = 400,

  /**
	 * This indicates that a named symbol was not found. Examples of symbols
	 * are global/constant variable names, texture names, and surface names.
	 */
  CUDA_ERROR_NOT_FOUND = 500,

  /**
	 * This indicates that asynchronous operations issued previously have not
	 * completed yet. This result is not actually an error, but must be indicated
	 * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
	 * may return this value include ::cuEventQuery() and ::cuStreamQuery().
	 */
  CUDA_ERROR_NOT_READY = 600,

  /**
	 * While executing a kernel, the device encountered a
	 * load or store instruction on an invalid memory address.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,

  /**
	 * This indicates that a launch did not occur because it did not have
	 * appropriate resources. This error usually indicates that the user has
	 * attempted to pass too many arguments to the device kernel, or the
	 * kernel launch specifies too many threads for the kernel's register
	 * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
	 * when a 32-bit int is expected) is equivalent to passing too many
	 * arguments and can also result in this error.
	 */
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,

  /**
	 * This indicates that the device kernel took too long to execute. This can
	 * only occur if timeouts are enabled - see the device attribute
	 * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
	 * context cannot be used (and must be destroyed similar to
	 * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
	 * this context are invalid and must be reconstructed if the program is to
	 * continue using CUDA.
	 */
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,

  /**
	 * This error indicates a kernel launch that uses an incompatible texturing
	 * mode.
	 */
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

  /**
	 * This error indicates that a call to ::cuCtxEnablePeerAccess() is
	 * trying to re-enable peer access to a context which has already
	 * had peer access to it enabled.
	 */
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

  /**
	 * This error indicates that ::cuCtxDisablePeerAccess() is
	 * trying to disable peer access which has not been enabled yet
	 * via ::cuCtxEnablePeerAccess().
	 */
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

  /**
	 * This error indicates that the primary context for the specified device
	 * has already been initialized.
	 */
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,

  /**
	 * This error indicates that the context current to the calling thread
	 * has been destroyed using ::cuCtxDestroy, or is a primary context which
	 * has not yet been initialized.
	 */
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,

  /**
	 * A device-side assert triggered during kernel execution. The context
	 * cannot be used anymore, and must be destroyed. All existing device
	 * memory allocations from this context are invalid and must be
	 * reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_ASSERT = 710,

  /**
	 * This error indicates that the hardware resources required to enable
	 * peer access have been exhausted for one or more of the devices
	 * passed to ::cuCtxEnablePeerAccess().
	 */
  CUDA_ERROR_TOO_MANY_PEERS = 711,

  /**
	 * This error indicates that the memory range passed to ::cuMemHostRegister()
	 * has already been registered.
	 */
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

  /**
	 * This error indicates that the pointer passed to ::cuMemHostUnregister()
	 * does not correspond to any currently registered memory region.
	 */
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,

  /**
	 * While executing a kernel, the device encountered a stack error.
	 * This can be due to stack corruption or exceeding the stack size limit.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,

  /**
	 * While executing a kernel, the device encountered an illegal instruction.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,

  /**
	 * While executing a kernel, the device encountered a load or store instruction
	 * on a memory address which is not aligned.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,

  /**
	 * While executing a kernel, the device encountered an instruction
	 * which can only operate on memory locations in certain address spaces
	 * (global, shared, or local), but was supplied a memory address not
	 * belonging to an allowed address space.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,

  /**
	 * While executing a kernel, the device program counter wrapped its address space.
	 * The context cannot be used, so it must be destroyed (and a new one should be created).
	 * All existing device memory allocations from this context are invalid
	 * and must be reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_INVALID_PC = 718,

  /**
	 * An exception occurred on the device while executing a kernel. Common
	 * causes include dereferencing an invalid device pointer and accessing
	 * out of bounds shared memory. The context cannot be used, so it must
	 * be destroyed (and a new one should be created). All existing device
	 * memory allocations from this context are invalid and must be
	 * reconstructed if the program is to continue using CUDA.
	 */
  CUDA_ERROR_LAUNCH_FAILED = 719,

  /**
	 * This error indicates that the attempted operation is not permitted.
	 */
  CUDA_ERROR_NOT_PERMITTED = 800,

  /**
	 * This error indicates that the attempted operation is not supported
	 * on the current system or device.
	 */
  CUDA_ERROR_NOT_SUPPORTED = 801,

  /**
	 * This indicates that an unknown internal error has occurred.
	 */
  CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef enum cudaError cudaError_t;

cudaError_t lastError;

cudaError_t
__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  __ESBMC_assert(count > 0, "Size to be allocated may not be less than zero");

  //verifCopyKind(dst,src,kind);

  char *cdst = (char *)dst;
  const char *csrc = (const char *)src;
  int numbytes = count / (sizeof(char));

  for(int i = 0; i < numbytes; i++)
    cdst[i] = csrc[i];

  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

template <class T1, class T2>
cudaError_t cudaMemcpy(T1 dst, T2 src, size_t count, enum cudaMemcpyKind kind)
{
  return __cudaMemcpy(dst, src, count, kind);
}

const char *cudaGetErrorString(cudaError_t error)
{
  char *erroReturn;

  switch(error)
  {
  case 0:
    return "CUDA_SUCCESS";
    break;
  case 1:
    return "CUDA_ERROR_INVALID_VALUE";
    break;
  case 2:
    return "CUDA_ERROR_OUT_OF_MEMORY";
    break;
  case 3:
    return "CUDA_ERROR_NOT_INITIALIZED";
    break;
  case 4:
    return "CUDA_ERROR_DEINITIALIZED";
    break;
  case 5:
    return "CUDA_ERROR_PROFILER_DISABLED";
    break;
  case 6:
    return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    break;
  case 7:
    return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    break;
  case 8:
    return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    break;
  case 100:
    return "CUDA_ERROR_NO_DEVICE";
    break;
  case 101:
    return "CUDA_ERROR_INVALID_DEVICE";
    break;
  case 200:
    return "CUDA_ERROR_INVALID_IMAGE";
    break;
  case 201:
    return "CUDA_ERROR_INVALID_CONTEXT";
    break;
  case 202:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    break;
  case 205:
    return "CUDA_ERROR_MAP_FAILED";
    break;
  case 206:
    return "CUDA_ERROR_UNMAP_FAILED";
    break;
  case 207:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";
    break;
  case 208:
    return "CUDA_ERROR_ALREADY_MAPPED";
    break;
  case 209:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    break;
  case 210:
    return "CUDA_ERROR_ALREADY_ACQUIRED";
    break;
  case 211:
    return "CUDA_ERROR_NOT_MAPPED";
    break;
  case 212:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    break;
  case 213:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    break;
  case 214:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";
    break;
  case 215:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    break;
  case 216:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    break;
  case 217:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    break;
  case 218:
    return "CUDA_ERROR_INVALID_PTX";
    break;
  case 219:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    break;
  case 300:
    return "CUDA_ERROR_INVALID_SOURCE";
    break;
  case 301:
    return "CUDA_ERROR_FILE_NOT_FOUND";
    break;
  case 302:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    break;
  case 303:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    break;
  case 304:
    return "CUDA_ERROR_OPERATING_SYSTEM";
    break;
  case 400:
    return "CUDA_ERROR_INVALID_HANDLE";
    break;
  case 500:
    return "CUDA_ERROR_NOT_FOUND";
    break;
  case 600:
    return "CUDA_ERROR_NOT_READY";
    break;
  case 700:
    return "CUDA_ERROR_ILLEGAL_ADDRESS";
    break;
  case 701:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    break;
  case 702:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";
    break;
  case 703:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    break;
  case 704:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    break;
  case 705:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    break;
  case 708:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    break;
  case 709:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    break;
  case 710:
    return "CUDA_ERROR_ASSERT";
    break;
  case 711:
    return "CUDA_ERROR_TOO_MANY_PEERS";
    break;
  case 712:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    break;
  case 713:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    break;
  case 714:
    return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    break;
  case 715:
    return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    break;
  case 716:
    return "CUDA_ERROR_MISALIGNED_ADDRESS";
    break;
  case 717:
    return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    break;
  case 718:
    return "CUDA_ERROR_INVALID_PC";
    break;
  case 719:
    return "CUDA_ERROR_LAUNCH_FAILED";
    break;
  case 800:
    return "CUDA_ERROR_NOT_PERMITTED";
    break;
  case 801:
    return "CUDA_ERROR_NOT_SUPPORTED";
    break;
  case 999:
    return "CUDA_ERROR_UNKNOWN";
    break;
  default:
    return "CUDA_SUCCESS";
  }
}

////////////////////////////////////////////////////////////////////////////
//!  Structure that represents the devices of CUDA.

//Struct to Device - OK
typedef struct cudaDevicesList
{
  int id;
  int active;
  struct cudaDeviceProp deviceProp;
  struct cudaDevicesList *prox;
} cudaDeviceList_t;

cudaDeviceList_t *cudaDeviceList = NULL;

//Insert a device
void cudaDeviceInsert(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  //Verifies that the device exists in the list
  while(auxDevice != NULL)
  {
    if(auxDevice->id == device)
    {
      //printf("\nDevice existing");
      //return 0;
    }
    auxDevice = auxDevice->prox;
  }

  //Insert new device
  cudaDeviceList_t *newCudaDevice;

  newCudaDevice = (cudaDeviceList_t *)malloc(sizeof(cudaDeviceList_t));
  if(newCudaDevice == NULL)
    exit(0);

  newCudaDevice->id = device;
  newCudaDevice->active = 0;
  //newCudaDevice->deviceProp.regsPerBlock = var; //Insert fields to deviceProp
  newCudaDevice->prox = NULL;

  if(cudaDeviceList == NULL)
  {
    cudaDeviceList = newCudaDevice;
  }
  else
  {
    newCudaDevice->prox = cudaDeviceList;
    cudaDeviceList = newCudaDevice;
  }

  //	return 1;
}

//Searching for a device in the devices list
void cudaPrintDevice()
{
  //printf("\n\n*** CUDA Device\n");

  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  {
    //printf("->Device: %d Active:%d\n",auxDevice->id,auxDevice->active);
    auxDevice = auxDevice->prox;
  }
}

//Searching for a device in the devices list
int searchCudaDevice(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  {
    if(auxDevice->id == device)
    {
      return 1;
    }
    else
    {
      auxDevice = auxDevice->prox;
    }
  }
  return 0;
}

//Checks whether the device is in use
int cudaDeviceActive(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  {
    if(auxDevice->id == device)
    {
      if(auxDevice->active == 1)
      {
        return 1;
      }
      else
      {
        return 0;
      }
    }
  }
  return 0;
}

//Start a device
int cudaDeviceStart(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  {
    if(auxDevice->id == device)
    {
      auxDevice->active = 1;
      return 1;
    }
    auxDevice = auxDevice->prox;
  }
  return 0;
}

// Choose a device to work
cudaError_t cudaSetDevice(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  { //Scroll through the list
    if(auxDevice->id == device)
    { //Checks if the device
      if(auxDevice->active == 1)
      {                                     //Verifies that the device is active
        return cudaErrorDeviceAlreadyInUse; //cudaErrorDeviceAlreadyInUse
        lastError = cudaErrorDeviceAlreadyInUse;
      }
      auxDevice->active = 1;
      lastError = cudaSuccess;
      return cudaSuccess;
    }
    else
      auxDevice = auxDevice->prox;
  }
  //If not found, return cudaErrorInvalidDevice
  lastError = cudaErrorInvalidDevice;
  return cudaErrorInvalidDevice;
}

// Returns the number of compute-capable devices.
cudaError_t cudaGetDeviceCount(int *count)
{
  /*
	cudaDeviceList_t *auxDevice = cudaDeviceList;
	int i;

	while(auxDevice!=NULL){
		i++;
		auxDevice = auxDevice->prox;
	}
	 */
  lastError = cudaSuccess;
  return cudaSuccess;
}

//Destroy all allocations and reset all state on the current device in the current process.
cudaError_t cudaDeviceReset()
{
  /*
	int tmp;
	threadsList_t *node;

	while(cudaThreadList != NULL){
		//		pthread_exit(cudaThreadList->thread);
		node = cudaThreadList;
		cudaThreadList = cudaThreadList->prox;
		free(node);
	}

	lastError = cudaSuccess;
	*/
  return cudaSuccess;
}

/*
 *    Memory Management
 */
cudaError_t cudaMemcpyToSymbol(
  const void *symbol,
  const void *src,
  size_t count,
  size_t offset __dv(0),
  enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
  cudaError_t out;

  out = __cudaMemcpy((void *)(symbol), (const void *)src, count, kind);

  lastError = out;
  return out;
}

cudaError_t cudaMemcpyFromSymbol(
  void *dst,
  const void *symbol,
  size_t count,
  size_t offset __dv(0),
  enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
  cudaError_t out;

  out = __cudaMemcpy((void *)dst, (const void *)(symbol), count, kind);

  lastError = out;
  return out;
}

extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

extern cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
  struct cudaDeviceProp deviceChosen;

  int out;

  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while(auxDevice != NULL)
  {
    if(auxDevice->id == device)
    {
      deviceChosen = auxDevice->deviceProp;
      prop = &deviceChosen;
      return cudaSuccess;
    }
    auxDevice = auxDevice->prox;
  }

  return CUDA_ERROR_INVALID_VALUE;
}

cudaError_t cudaGetLastError()
{
  return lastError;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceSynchronize(void);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

extern __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr);

extern __host__ cudaError_t CUDARTAPI cudaThreadExit(void);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaPeekAtLastError(void);

extern __host__ __cudart_builtin__ const char *CUDARTAPI
cudaGetErrorName(cudaError_t error);

extern __host__ __cudart_builtin__ const char *CUDARTAPI
cudaGetErrorString(cudaError_t error);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);

extern __host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDevice(int *device);

extern __host__ cudaError_t CUDARTAPI
cudaSetValidDevices(int *device_arr, int len);

extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);

extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventDestroy(cudaEvent_t event);

extern __host__ cudaError_t CUDARTAPI
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

extern __host__ cudaError_t CUDARTAPI
cudaSetupArgument(const void *arg, size_t size, size_t offset);

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d);

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size);

extern __host__ cudaError_t CUDARTAPI
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);

extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaHostAlloc(void **pHost, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaHostRegister(void *ptr, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaHostGetFlags(unsigned int *pFlags, void *pHost);

extern __host__ cudaError_t CUDARTAPI
cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent);

extern __host__ cudaError_t CUDARTAPI
cudaMemcpy3D(const struct cudaMemcpy3DParms *p);

extern __host__ cudaError_t CUDARTAPI
cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);

extern __host__ cudaError_t CUDARTAPI
cudaMemGetInfo(size_t *free, size_t *total);

extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(
  void *dst,
  int dstDevice,
  const void *src,
  int srcDevice,
  size_t count);

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(
  void *dst,
  size_t dpitch,
  const void *src,
  size_t spitch,
  size_t width,
  size_t height,
  enum cudaMemcpyKind kind);

extern __host__ cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count);

extern __host__ cudaError_t CUDARTAPI cudaMemset2D(
  void *devPtr,
  size_t pitch,
  int value,
  size_t width,
  size_t height);

extern __host__ cudaError_t CUDARTAPI cudaMemset3D(
  struct cudaPitchedPtr pitchedDevPtr,
  int value,
  struct cudaExtent extent);

extern __host__ cudaError_t CUDARTAPI
cudaGetSymbolAddress(void **devPtr, const void *symbol);

extern __host__ cudaError_t CUDARTAPI
cudaGetSymbolSize(size_t *size, const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(
  struct cudaPointerAttributes *attributes,
  const void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceDisablePeerAccess(int peerDevice);

extern __host__ cudaError_t CUDARTAPI cudaBindTexture(
  size_t *offset,
  const struct textureReference *texref,
  const void *devPtr,
  const struct cudaChannelFormatDesc *desc,
  size_t size __dv(UINT_MAX));

extern __host__ cudaError_t CUDARTAPI cudaBindTexture2D(
  size_t *offset,
  const struct textureReference *texref,
  const void *devPtr,
  const struct cudaChannelFormatDesc *desc,
  size_t width,
  size_t height,
  size_t pitch);

extern __host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference *texref);

extern __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
  size_t *offset,
  const struct textureReference *texref);

extern __host__ cudaError_t CUDARTAPI cudaGetTextureReference(
  const struct textureReference **texref,
  const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(
  const struct surfaceReference **surfref,
  const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaRuntimeGetVersion(int *runtimeVersion);

#undef __dv

#endif /* cuda_runtime_api.h */
