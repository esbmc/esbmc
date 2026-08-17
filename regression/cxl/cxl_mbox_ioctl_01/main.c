// CXL mailbox IOCTL command table lookup: unknown opcodes, oversized
// payloads and missing buffers are all rejected before the device is told
// anything.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_dev cxld;
  char payload[64];

  /* An opcode absent from the command table is rejected outright. */
  assert(cxl_mbox_cmd_index(0xFFFF) < 0);
  assert(cxl_mailbox_ioctl(&cxld, 0xFFFF, payload, sizeof(payload)) == -ENOTTY);

  /* One that is present resolves to a table slot. */
  assert(cxl_mbox_cmd_index(CXL_MBOX_OP_GET_STATUS) >= 0);

  /* A payload larger than the mailbox accepts never reaches the device. */
  assert(
    cxl_mailbox_ioctl(
      &cxld,
      CXL_MBOX_OP_GET_STATUS,
      payload,
      CXL_MBOX_IOCTL_MAX_PAYLOAD + 1) == -EINVAL);

  /* A non-zero length with no buffer is rejected. */
  assert(cxl_mailbox_ioctl(&cxld, CXL_MBOX_OP_GET_STATUS, NULL, 8) == -EINVAL);

  /* A well-formed call either runs or reports the command disabled on
     this device — never anything else. */
  int ret =
    cxl_mailbox_ioctl(&cxld, CXL_MBOX_OP_GET_STATUS, payload, sizeof(payload));
  assert(ret == 0 || ret == -ENOTTY);

  return 0;
}
