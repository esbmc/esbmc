// CXL ACPI driver that sizes its interleave-target array from the decoded
// ways, then fills it using the raw CFMWS field.
// Expected: VERIFICATION FAILED (driver bug: out-of-bounds write, CWE-787)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

extern unsigned char __VERIFIER_nondet_uchar(void);

#define CFMWS_HDR_LEN 36

int main()
{
  struct acpi_cedt_cfmws w;
  unsigned int ways;
  u32 *targets;

  memset(&w, 0, sizeof(w));
  w.interleave_arithmetic = ACPI_CEDT_CFMWS_ARITHMETIC_MODULO;
  w.base_hpa = 4UL * CXL_SZ_256M;
  w.window_size = 8UL * CXL_SZ_256M;
  w.interleave_ways = __VERIFIER_nondet_uchar();
  w.length = CFMWS_HDR_LEN + 4 * CXL_CFMWS_MAX_WAYS;

  if (acpi_cedt_parse_cfmws(&w, &ways))
    return 0;

  /* Sized from the decoded count, as cxl_root_decoder_alloc() does. */
  targets = kmalloc(ways * sizeof(*targets), GFP_KERNEL);
  if (!targets)
    return 0;

  /*
   * BUG: filled using the raw field. interleave_ways is an EIW *encoding*,
   * not a count -- and the two are not even ordered the same way. EIW 8
   * decodes to 3 ways, so the array holds three entries and this writes
   * eight. The encodings that mislead are the 3/6/12 ones added by a later
   * ECN; the power-of-two ones a machine usually boots with happen to be
   * harmless here, which is what lets it survive.
   */
  for (u8 i = 0; i < w.interleave_ways; i++)
    targets[i] = i;

  kfree(targets);
  return 0;
}
