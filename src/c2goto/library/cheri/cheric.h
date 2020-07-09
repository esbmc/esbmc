#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#undef cheri_ptr

#define __capability

static __inline void *__capability cheri_ptr(const void *ptr, size_t len)
{
__ESBMC_HIDE:;
  // zero-length capability is allowed
  __ESBMC_assert(len >= 0, "len must be greater or equal than zero");
  // ensure that ptr is valid and is in bounds
  __ESBMC_assert(ptr != NULL, "ptr must not point to NULL");
  __ESBMC_assert_object_size(ptr, len);
  // declare the capability pointer and keep track of its bounds
  // within our symbolic execution engine
  void *cap_ptr;
  __ESBMC_cheri_bounds_set(cap_ptr, ptr, len);
  return cap_ptr;
}
