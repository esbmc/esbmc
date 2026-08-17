// DAX probe that treats an inclusive HPA range as exclusive when sizing the
// device.
// Expected: VERIFICATION FAILED (driver bug: region length one byte short)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define DAX_PMD_SIZE 0x200000ULL

int main()
{
  struct cxl_region_params p;
  u64 start, end;

  memset(&p, 0, sizeof(p));
  p.state = CXL_CONFIG_COMMIT;
  p.nr_targets = 1;
  p.res_start = 0x100000000ULL;
  p.res_end = 0x100000000ULL + 8 * DAX_PMD_SIZE - 1;

  if (cxl_dax_region_alloc(&p, &start, &end))
    return 0;

  /*
   * BUG: struct resource is inclusive at both ends, so the length is
   * end - start + 1. Dropping the +1 loses the final byte -- and because the
   * region is PMD-aligned, the shortfall also breaks the alignment the DAX
   * device is created with, which is the part that actually bites.
   */
  u64 len = end - start;

  assert(len % DAX_PMD_SIZE == 0);
  return 0;
}
