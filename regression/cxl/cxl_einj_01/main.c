// ACPI EINJ error injection to a CXL port (drivers/acpi/apei/einj-cxl.c).
//
// einj_cxl_inject_rch_error() refuses any type that is not one of the six
// CXL protocol error types -- EINJ carries plenty of others, and injecting a
// non-CXL type through the CXL path would target the wrong thing.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

#define RCRB 0x10000000ULL

int main()
{
  /* Each of the six CXL types is accepted. */
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_CACHE_CORRECTABLE));
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_CACHE_UNCORRECTABLE));
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_CACHE_FATAL));
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_MEM_CORRECTABLE));
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_MEM_UNCORRECTABLE));
  assert(einj_is_cxl_error_type(ACPI_EINJ_CXL_MEM_FATAL));

  /* A processor or memory error type is not a CXL one. */
  assert(!einj_is_cxl_error_type(0x1));
  assert(!einj_is_cxl_error_type(0x8));
  assert(!einj_is_cxl_error_type(0));

  assert(einj_cxl_inject_rch_error(RCRB, ACPI_EINJ_CXL_MEM_FATAL) == 0);
  assert(einj_cxl_inject_rch_error(RCRB, 0x1) == -EINVAL);
  /* No RCRB means no downstream port to inject into. */
  assert(einj_cxl_inject_rch_error(0, ACPI_EINJ_CXL_MEM_FATAL) == -EINVAL);

  /* Acceptance implies both obligations were met. */
  u64 t = (u64)__VERIFIER_nondet_ulong();
  u64 rcrb = (u64)__VERIFIER_nondet_ulong();
  if (einj_cxl_inject_rch_error(rcrb, t) == 0)
  {
    assert(einj_is_cxl_error_type(t));
    assert(rcrb != 0);
  }

  return 0;
}
