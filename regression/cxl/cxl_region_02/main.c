// CXL overlapping region targets: the driver commits a second region
// without checking it against the ones already live.
// Expected: VERIFICATION FAILED (driver bug: missing overlap check)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define DRIVER_MAX_REGIONS 2
static struct cxl_region driver_regions[DRIVER_MAX_REGIONS];
static unsigned int driver_region_count;

/*
 * BUG: a correct driver rejects a region whose host physical address
 * range intersects one already committed — that is what
 * cxl_region_overlaps() is for. This one validates only the interleave
 * geometry, so two decoders end up claiming the same HPA.
 */
static int driver_add_region(
  resource_size_t start,
  resource_size_t size,
  unsigned int ways,
  unsigned int granularity)
{
  if (driver_region_count >= DRIVER_MAX_REGIONS)
    return -1;

  struct cxl_region *r = &driver_regions[driver_region_count];
  r->start = start;
  r->size = size;
  if (cxl_region_config(r, ways, granularity) != 0)
    return -1;

  driver_region_count++;
  return 0;
}

int main()
{
  assert(driver_add_region(0x40000000, 0x10000, 4, 256) == 0);

  /* Overlaps the first region by half its length. */
  assert(driver_add_region(0x40008000, 0x10000, 4, 256) == 0);

  __ESBMC_assert(
    !cxl_region_overlaps(&driver_regions[0], &driver_regions[1]),
    "committed CXL regions must not overlap");
  return 0;
}
