// CXL mailbox command with a reply buffer smaller than a register, plus the
// error-injection counters.
//
// The mailbox fills payload_out on success only, and fills at most what
// payload_out_size offered. GET_STATUS returns two bytes; a driver asking for
// two must get two written, not four.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  struct cxl_mailbox_cmd cmd;
  unsigned char reply[2];
  unsigned char canary = 0xA5;

  memset(&pdev, 0, sizeof(pdev));
  memset(&cxld, 0, sizeof(cxld));
  cxld.regs = pci_iomap(&pdev, 0, 4096);
  if (!cxld.regs)
    return 0;

  memset(&cmd, 0, sizeof(cmd));
  cmd.opcode = CXL_MBOX_OP_GET_STATUS;
  cmd.payload_out = reply;
  cmd.payload_out_size = sizeof(reply);

  if (cxl_mailbox_send_cmd(&cxld, &cmd) == 0)
    assert(cmd.status == 0);
  assert(canary == 0xA5);

  /* Error injection is fallible hardware too; the counters only move when it
     reports success. */
  int c0, n0, f0, c1, n1, f1;
  if (cxl_err_get_count(&cxld, &c0, &n0, &f0) == 0 &&
      cxl_err_inject(&cxld, CXL_ERR_CORRECTABLE) == 0 &&
      cxl_err_get_count(&cxld, &c1, &n1, &f1) == 0)
  {
    assert(c1 == c0 + 1);
    assert(n1 == n0);
    assert(f1 == f0);
  }

  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
