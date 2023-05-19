#include "vector_types.h"

/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicAdd(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicAdd(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicExch(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicExch(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ float __fAtomicExch(float *address, float val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicMin(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicMin(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicMax(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicMax(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicInc(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicDec(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicAnd(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicAnd(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicOr(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicOr(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicXor(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicXor(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicCAS(int *address, int compare, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val);

/***********/

extern __device__ int __iAtomicAdd(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicAdd(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicExch(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;

  old_value = *address;
  *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicExch(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;

  old_value = *address;
  *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ float __fAtomicExch(float *address, float val)
{
  __ESBMC_atomic_begin();
  float old_value;

  old_value = *address;
  *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicMin(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicMin(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicMax(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicMax(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicInc(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;

  old_value = *address;

  //	((old >= val) ? 0 : (old+1))
  //	(old_value >= val) ? (*address = 0) : (*address = (old_value+1));

  if(old_value >= val)
    *address = 0;
  else
    *address = (old_value + 1);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicDec(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;

  old_value = *address;

  //	(((old == 0) | (old > val)) ? val : (old-1) )
  //	((old_value == val) | (old_value > val)) ? (*address = val) : (*address = (old_value-1));

  if((old_value == val) | (old_value > val))
    *address = val;
  else
    *address = (old_value - 1);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicAnd(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;
  old_value = *address;

  *address = (old_value & val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicAnd(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;
  old_value = *address;

  *address = (old_value & val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicOr(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;
  old_value = *address;

  *address = (old_value | val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicOr(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;
  old_value = *address;

  *address = (old_value | val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicXor(int *address, int val)
{
  __ESBMC_atomic_begin();
  int old_value;
  old_value = *address;

  *address = (old_value ^ val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicXor(unsigned int *address, unsigned int val)
{
  __ESBMC_atomic_begin();
  int old_value;
  old_value = *address;

  *address = (old_value ^ val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ int __iAtomicCAS(int *address, int compare, int val)
{
  __ESBMC_atomic_begin();
  int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned int
__uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  __ESBMC_atomic_begin();
  unsigned int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  __ESBMC_atomic_end();
  return old_value;
}

/***********/

static __inline__ __device__ int atomicAdd(int *address, int val)
{
  return __iAtomicAdd(address, val);
}

static __inline__ __device__ unsigned int
atomicAdd(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, val);
}

static __inline__ __device__ int atomicSub(int *address, int val)
{
  return __iAtomicAdd(address, -(int)val);
}

static __inline__ __device__ unsigned int
atomicSub(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, -(unsigned int)val);
}

static __inline__ __device__ int atomicExch(int *address, int val)
{
  return __iAtomicExch(address, val);
}

static __inline__ __device__ unsigned int
atomicExch(unsigned int *address, unsigned int val)
{
  return __uAtomicExch(address, val);
}

static __inline__ __device__ float atomicExch(float *address, float val)
{
  return __fAtomicExch(address, val);
}

static __inline__ __device__ int atomicMin(int *address, int val)
{
  return __iAtomicMin(address, val);
}

static __inline__ __device__ unsigned int
atomicMin(unsigned int *address, unsigned int val)
{
  return __uAtomicMin(address, val);
}

static __inline__ __device__ int atomicMax(int *address, int val)
{
  return __iAtomicMax(address, val);
}

static __inline__ __device__ unsigned int
atomicMax(unsigned int *address, unsigned int val)
{
  return __uAtomicMax(address, val);
}

static __inline__ __device__ unsigned int
atomicInc(unsigned int *address, unsigned int val)
{
  return __uAtomicInc(address, val);
}

static __inline__ __device__ unsigned int
atomicDec(unsigned int *address, unsigned int val)
{
  return __uAtomicDec(address, val);
}

static __inline__ __device__ int atomicAnd(int *address, int val)
{
  return __iAtomicAnd(address, val);
}

static __inline__ __device__ unsigned int
atomicAnd(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd(address, val);
}

static __inline__ __device__ int atomicOr(int *address, int val)
{
  return __iAtomicOr(address, val);
}

static __inline__ __device__ unsigned int
atomicOr(unsigned int *address, unsigned int val)
{
  return __uAtomicOr(address, val);
}

static __inline__ __device__ int atomicXor(int *address, int val)
{
  return __iAtomicXor(address, val);
}

static __inline__ __device__ unsigned int
atomicXor(unsigned int *address, unsigned int val)
{
  return __uAtomicXor(address, val);
}

static __inline__ __device__ int atomicCAS(int *address, int compare, int val)
{
  return __iAtomicCAS(address, compare, val);
}

static __inline__ __device__ unsigned int
atomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  return __uAtomicCAS(address, compare, val);
}

//sm_12_atomic_functions.h

/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int
__ullAtomicAdd(unsigned long long int *address, unsigned long long int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int
__ullAtomicExch(unsigned long long int *address, unsigned long long int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int __ullAtomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val);

/*DEVICE_BUILTIN*/
extern __device__ int __any(int cond);
/*DEVICE_BUILTIN*/
extern __device__ int __all(int cond);

/***********/

extern __device__ unsigned long long int
__ullAtomicAdd(unsigned long long int *address, unsigned long long int val)
{
  __ESBMC_atomic_begin();
  unsigned long long int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned long long int
__ullAtomicExch(unsigned long long int *address, unsigned long long int val)
{
  __ESBMC_atomic_begin();
  unsigned long long int old_value;

  old_value = *address;
  *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ unsigned long long int __ullAtomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val)
{
  __ESBMC_atomic_begin();
  unsigned long long int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  __ESBMC_atomic_end();
  return old_value;
}

/*
extern __device__ int                    __any(int cond){
}

extern __device__ int                    __all(int cond){
}
*/

/***********/

static __inline__ __device__ unsigned long long int
atomicAdd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAdd(address, val);
}

static __inline__ __device__ unsigned long long int
atomicExch(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicExch(address, val);
}

static __inline__ __device__ unsigned long long int atomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val)
{
  return __ullAtomicCAS(address, compare, val);
}

static __inline__ __device__ bool any(bool cond)
{
  return (bool)__any((int)cond);
}

static __inline__ __device__ bool all(bool cond)
{
  return (bool)__all((int)cond);
}

//sm_20_atomic_functions.h

/*DEVICE_BUILTIN*/
extern __device__ float __fAtomicAdd(float *address, float val);

/***********/

extern __device__ float __fAtomicAdd(float *address, float val)
{
  __ESBMC_atomic_begin();
  float old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  __ESBMC_atomic_end();
  return old_value;
}

/***********/

static __inline__ __device__ float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}

//sm_32_atomic_functions.h

extern __device__ __device_builtin__ long long
__illAtomicMin(long long *address, long long val);
extern __device__ __device_builtin__ long long
__illAtomicMax(long long *address, long long val);
extern __device__ __device_builtin__ unsigned long long
__ullAtomicMin(unsigned long long *address, unsigned long long val);
extern __device__ __device_builtin__ unsigned long long
__ullAtomicMax(unsigned long long *address, unsigned long long val);
extern __device__ __device_builtin__ unsigned long long int
__ullAtomicAnd(unsigned long long int *address, unsigned long long int val);
extern __device__ __device_builtin__ unsigned long long
__ullAtomicOr(unsigned long long *address, unsigned long long val);
extern __device__ __device_builtin__ unsigned long long
__ullAtomicXor(unsigned long long *address, unsigned long long val);

/***********/

extern __device__ __device_builtin__ long long
__illAtomicMin(long long *address, long long val)
{
  __ESBMC_atomic_begin();
  long long old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ long long
__illAtomicMax(long long *address, long long val)
{
  __ESBMC_atomic_begin();
  long long old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ unsigned long long
__ullAtomicMin(unsigned long long *address, unsigned long long val)
{
  __ESBMC_atomic_begin();
  unsigned long long old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ unsigned long long
__ullAtomicMax(unsigned long long *address, unsigned long long val)
{
  __ESBMC_atomic_begin();
  unsigned long long old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ unsigned long long int
__ullAtomicAnd(unsigned long long int *address, unsigned long long int val)
{
  __ESBMC_atomic_begin();
  unsigned long long int old_value;
  old_value = *address;

  *address = (old_value & val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ unsigned long long
__ullAtomicOr(unsigned long long *address, unsigned long long val)
{
  __ESBMC_atomic_begin();
  unsigned long long old_value;
  old_value = *address;

  *address = (old_value | val);

  __ESBMC_atomic_end();
  return old_value;
}

extern __device__ __device_builtin__ unsigned long long
__ullAtomicXor(unsigned long long *address, unsigned long long val)
{
  __ESBMC_atomic_begin();
  unsigned long long old_value;
  old_value = *address;

  *address = (old_value ^ val);

  __ESBMC_atomic_end();
  return old_value;
}

/***********/

static __inline__ __device__ long long
atomicMin(long long *address, long long val)
{
  return __illAtomicMin(address, val);
}

static __inline__ __device__ long long
atomicMax(long long *address, long long val)
{
  return __illAtomicMax(address, val);
}

static __inline__ __device__ unsigned long long
atomicMin(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMin(address, val);
}

static __inline__ __device__ unsigned long long
atomicMax(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMax(address, val);
}

static __inline__ __device__ unsigned long long int
atomicAnd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAnd(address, val);
}

static __inline__ __device__ unsigned long long
atomicOr(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicOr(address, val);
}

static __inline__ __device__ unsigned long long
atomicXor(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicXor(address, val);
}
