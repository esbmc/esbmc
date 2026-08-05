// CXL HDM decoder misaligned region detection test.
// Tests that the driver rejects HDM decoder configurations with
// base addresses that are NOT 4KB-aligned.
// Expected: VERIFICATION FAILED (driver bug: allows misaligned address)
//
// Based on Linux kernel drivers/cxl/cxl_core.c::cxl_add_hdm_decoder()
// where real drivers validate alignment before programming decoders.

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>

#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

/* Simulated decoder tracking */
static int decoder_count = 0;

/*
 * BUG: This driver does NOT validate 4KB alignment on the base address.
 * A correct implementation must return -EINVAL for misaligned addresses.
 */
int cxl_setup_hdm_decoders(struct cxl_dev *cxld,
                           const struct cxl_region *region)
{
  (void)cxld;
  (void)region;

  /* BUG: No alignment check! */

  if (decoder_count >= CXL_HDM_DECODER_MAX)
    return -1;

  decoder_count++;
  return 0;
}

int main()
{
  struct cxl_dev test_cxld;
  struct cxl_region region;

  test_cxld.regs = (void *)0xFED00000;

  /* Setup decoder 0: aligned base address */
  region.start = 0;
  region.size = 256 * 1024 * 1024;
  region.granularity = 1;

  int ret = cxl_setup_hdm_decoders(&test_cxld, &region);
  assert(ret == 0);

  /*
   * BUG: Setup decoder 1 with a misaligned base address (128 bytes off).
   * The CXL 2.0 spec requires 4KB alignment for HDM decoder base addresses.
   * The driver should reject this but doesn't.
   */
  region.start = 0x80000080;  /* 2GB + 128 bytes — NOT 4KB-aligned */
  ret = cxl_setup_hdm_decoders(&test_cxld, &region);
  assert(ret == 0);  /* Bug: should return -EINVAL */

  /*
   * The invariant: all decoder base addresses must be 4KB-aligned.
   * Since the buggy driver allows misalignment, this assertion fails.
   */
  __ESBMC_assert((0x80000080 % CXL_HDM_ALIGNMENT) == 0,
                 "HDM decoder base address not 4KB-aligned");
}
