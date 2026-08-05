// CXL mailbox IOCTL with an unsupported payload size: the driver bounds the
// user-supplied length against what the mailbox accepts, not against its own
// staging buffer.
// Expected: VERIFICATION FAILED (driver bug: unchecked payload size)

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

/* The driver stages every command payload through one fixed buffer. */
static char driver_staging[256];

/*
 * BUG: cxl_mailbox_ioctl() accepts payloads up to
 * CXL_MBOX_IOCTL_MAX_PAYLOAD (4096) — that is the mailbox limit, not the
 * driver's. Having passed that check the driver treats the length as safe
 * for driver_staging, which holds only 256 bytes.
 */
static int driver_send(struct cxl_dev *cxld, u16 opcode, u32 user_size)
{
  int ret = cxl_mailbox_ioctl(cxld, opcode, driver_staging, user_size);
  if (ret != 0)
    return ret;

  memset(driver_staging, 0, user_size);
  return 0;
}

int main()
{
  struct cxl_dev cxld;

  driver_send(&cxld, CXL_MBOX_OP_GET_STATUS, 1024);
  return 0;
}
