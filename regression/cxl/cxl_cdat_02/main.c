// CDAT reader that checksums the length the device reported back, rather than
// the length it allocated for.
// Expected: VERIFICATION FAILED (driver bug: out-of-bounds read, CWE-125)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

extern unsigned char __VERIFIER_nondet_uchar(void);

/*
 * Stands in for cxl_cdat_read_table(), which takes &length and may write back
 * a different one. read_cdat_data() saves the original as table_length and
 * warns when they differ -- "discarding trailing data" -- but the warning is
 * all it does.
 */
static void cdat_read_table(unsigned char *buf, size_t *length)
{
  (void)buf;
  *length = (size_t)__VERIFIER_nondet_uchar();
}

int main()
{
  size_t length = (size_t)__VERIFIER_nondet_uchar();
  __ESBMC_assume(length > 0 && length <= 64);

  unsigned char *buf = kmalloc(length, GFP_KERNEL);
  if (!buf)
    return 0;
  memset(buf, 0, length);

  size_t table_length = length;
  cdat_read_table(buf, &length);

  /*
   * BUG: the readback length is used unchecked. The name table_length and the
   * warning text both suggest the mismatch was considered -- but "discarding
   * trailing data" only describes length < table_length. Nothing rules out
   * the other direction, and the buffer was sized for the original.
   */
  if (table_length != length)
  {
    /* warn only, as the driver does */
  }

  unsigned char sum = cdat_checksum(buf, length);
  assert(sum == 0);

  kfree(buf);
  return 0;
}
