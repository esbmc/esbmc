// DAX device registration over a committed CXL region (drivers/dax/cxl.c).
//
// cxl_dax_region_probe() builds a dax_region from the region's HPA range and
// hands its length to devm_create_dev_dax(). The length is the inclusive
// range's, so it must be computed as end - start + 1.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

#define DAX_PMD_SIZE 0x200000ULL /* PMD_SIZE on x86-64 */

int main()
{
  struct cxl_region_params p;
  u64 start, end;

  memset(&p, 0, sizeof(p));
  p.state = CXL_CONFIG_COMMIT;
  p.nr_targets = 1;
  /* A DAX-kmem region is PMD-aligned at both ends. */
  p.res_start = 0x100000000ULL;
  p.res_end = 0x100000000ULL + 8 * DAX_PMD_SIZE - 1;

  int rc = cxl_dax_region_alloc(&p, &start, &end);
  assert(rc == 0);

  u64 len = cxl_range_len(start, end);
  assert(len == 8 * DAX_PMD_SIZE);
  assert(start % DAX_PMD_SIZE == 0);
  assert(len % DAX_PMD_SIZE == 0);

  /* The last byte of the region is inside it; the byte after is not. */
  assert(start + len - 1 == end);

  /* A probe on an unregistered region gets nothing to map. */
  p.state = CXL_CONFIG_ACTIVE;
  assert(cxl_dax_region_alloc(&p, &start, &end) == -ENXIO);

  return 0;
}
