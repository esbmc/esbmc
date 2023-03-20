
#if !defined(__SM_20_ATOMIC_FUNCTIONS_H__)
#define __SM_20_ATOMIC_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "host_defines.h"

#define __location__(device) void
#define __device__ __location__(device)

/*DEVICE_BUILTIN*/
extern __device__ float __fAtomicAdd(float *address, float val);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ float __fAtomicAdd(float *address, float val)
{
  float old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  return old_value;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __device__ float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}

#endif /* !__SM_20_ATOMIC_FUNCTIONS_H__ */
