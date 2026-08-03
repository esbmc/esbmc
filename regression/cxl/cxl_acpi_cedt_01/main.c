// ACPI CEDT CFMWS window validation, against the real rules in
// cxl_acpi_cfmws_verify() (drivers/cxl/acpi.c).
//
// The interleave-ways field is an *encoding* (EIW), not a count: 0-4 mean
// 1,2,4,8,16 ways and 8-10 mean 3,6,12. Everything else is invalid, and a
// window is only usable if it is 256MB-aligned and long enough to carry one
// interleave target per way.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned char __VERIFIER_nondet_uchar(void);
extern unsigned long __VERIFIER_nondet_ulong(void);

#define CFMWS_HDR_LEN 36

int main()
{
  unsigned int ways;

  /* The encoding is total in one direction only: every accepted EIW yields a
     ways count that is either a power of two up to 16, or 3/6/12. */
  u8 eiw = __VERIFIER_nondet_uchar();
  if (eiw_to_ways(eiw, &ways) == 0)
  {
    assert(eiw <= 4 || (eiw >= 8 && eiw <= 10));
    assert(ways >= 1 && ways <= CXL_CFMWS_MAX_WAYS);
    if (eiw <= 4)
      assert((ways & (ways - 1)) == 0);
    else
      assert(ways == 3 || ways == 6 || ways == 12);
  }
  else
    assert(eiw == 5 || eiw == 6 || eiw == 7 || eiw > 10);

  /* A well-formed entry passes. */
  struct acpi_cedt_cfmws w;
  memset(&w, 0, sizeof(w));
  w.interleave_arithmetic = ACPI_CEDT_CFMWS_ARITHMETIC_MODULO;
  w.base_hpa = 4UL * CXL_SZ_256M;
  w.window_size = 8UL * CXL_SZ_256M;
  w.interleave_ways = 2; /* EIW 2 -> 4 ways */
  w.length = CFMWS_HDR_LEN + 4 * 4;
  assert(acpi_cedt_parse_cfmws(&w, &ways) == 0);
  assert(ways == 4);

  /* Whatever firmware supplies, acceptance implies every rule held. */
  struct acpi_cedt_cfmws nd;
  nd.interleave_arithmetic = __VERIFIER_nondet_uchar();
  nd.base_hpa = __VERIFIER_nondet_ulong();
  nd.window_size = __VERIFIER_nondet_ulong();
  nd.interleave_ways = __VERIFIER_nondet_uchar();
  nd.length = (u32)__VERIFIER_nondet_ulong();
  nd.granularity = 0;
  nd.restrictions = 0;

  if (acpi_cedt_parse_cfmws(&nd, &ways) == 0)
  {
    assert(nd.interleave_arithmetic == ACPI_CEDT_CFMWS_ARITHMETIC_MODULO ||
           nd.interleave_arithmetic == ACPI_CEDT_CFMWS_ARITHMETIC_XOR);
    assert(nd.base_hpa % CXL_SZ_256M == 0);
    assert(nd.window_size % CXL_SZ_256M == 0);
    assert(ways >= 1 && ways <= CXL_CFMWS_MAX_WAYS);
    /* Long enough to hold the targets the driver will go on to read. */
    assert(nd.length >= CFMWS_HDR_LEN + 4 * ways);
  }

  return 0;
}
