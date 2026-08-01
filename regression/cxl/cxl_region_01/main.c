// CXL region interleave configuration: valid geometries are accepted,
// unencodable ones rejected, and overlap detection is exact at the edges.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_region region;
  region.start = 0;
  region.size = 16384; /* a whole number of 4 x 256-byte stripes */
  region.ways = 0;
  region.granularity = 0;

  assert(cxl_region_config(&region, 4, 256) == 0);
  assert(region.ways == 4);
  assert(region.granularity == 256);

  /* Non-power-of-two ways is unencodable (ways_to_eiw()). */
  assert(cxl_region_config(&region, 3, 256) == -1);

  /* Granularity outside [256, 16384] is unencodable (granularity_to_eig()). */
  assert(cxl_region_config(&region, 4, 128) == -1);
  assert(cxl_region_config(&region, 4, 32768) == -1);

  /* A size that does not divide across the interleave set leaves one
     position owning a partial stripe. */
  region.size = 16384 + 1;
  assert(cxl_region_config(&region, 4, 256) == -1);

  /* A rejected configuration must leave the committed geometry intact. */
  assert(region.ways == 4);
  assert(region.granularity == 256);

  /* Overlap is half-open: touching regions are disjoint ... */
  struct cxl_region a;
  struct cxl_region b;
  a.start = 0x10000;
  a.size = 0x1000;
  b.start = 0x11000;
  b.size = 0x1000;
  assert(cxl_region_overlaps(&a, &b) == 0);

  /* ... but a one-byte intrusion is not. */
  b.start = 0x10FFF;
  assert(cxl_region_overlaps(&a, &b) != 0);

  return 0;
}
