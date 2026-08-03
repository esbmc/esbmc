// CXL mailbox caller that reads the reply buffer without checking whether the
// command succeeded.
// Expected: VERIFICATION FAILED (driver bug: reply consumed on the error
// path, where the mailbox never wrote it)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_STATUS_SENTINEL 0xFFFFFFFFU

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  struct cxl_mailbox_cmd cmd;
  uint32_t reply = CXL_STATUS_SENTINEL;

  memset(&pdev, 0, sizeof(pdev));
  memset(&cxld, 0, sizeof(cxld));
  cxld.regs = pci_iomap(&pdev, 0, 4096);
  if (!cxld.regs)
    return 0;

  memset(&cmd, 0, sizeof(cmd));
  cmd.opcode = CXL_MBOX_OP_GET_STATUS;
  cmd.payload_out = &reply;
  cmd.payload_out_size = sizeof(reply);

  /*
   * BUG: the return code is dropped. On failure the mailbox leaves
   * payload_out exactly as the caller left it and reports the reason in
   * cmd.status -- so the buffer still holds the caller's own sentinel, and
   * this reads it back as if the device had answered.
   */
  cxl_mailbox_send_cmd(&cxld, &cmd);
  assert(reply != CXL_STATUS_SENTINEL);

  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
