// CXL mailbox doorbell protocol as a Linux RV automaton, checked by ESBMC.
//
// cxl_mbox.dot is a deterministic automaton in the format Linux RV uses
// (kernel/trace/rv/). rv_cxl_mbox.h was produced from it by the kernel's own
// tools/verification/rvgen dot2c, then by scripts/rv2c.py -- so the same .dot
// could generate a live kernel monitor and this static check.
//
// The protocol is the one in the comment at drivers/cxl/pci.c:210, quoting
// CXL 2.0 §8.2.8.4:
//   1. verify the doorbell is clear      -- only legal from mbox_idle
//   4. write MB Control to set doorbell  -- mbox_submit
//   5. poll for the doorbell to clear, or time out
//
// The obligation is step 1: no command may be submitted while one is in
// flight. cxl_pci_mbox_wait_for_doorbell() (pci.c:57) is what makes
// mbox_timeout reachable, so a driver without that timeout has no way back to
// mbox_idle when the device stalls.
//
// This is what RV cannot do: RV observes the sequences a running kernel
// happened to produce. Here the driver's choices are symbolic, so every
// sequence it could produce is covered at once.
// Expected: VERIFICATION SUCCESSFUL

#include <assert.h>
#include "rv_cxl_mbox.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

/* A driver that observes the protocol: one command in flight at a time, and
   the wait always ends -- either the device answers or the timeout fires. */
int main(void)
{
  rv_cxl_mbox_reset();

  for (int cmd = 0; cmd < 3; cmd++)
  {
    rv_cxl_mbox_event(mbox_submit);

    /* cxl_pci_mbox_wait_for_doorbell(): poll, then give up. Whichever way it
       goes, the mailbox returns to idle. */
    if (__VERIFIER_nondet_uint() & 1u)
      rv_cxl_mbox_event(doorbell_clear);
    else
      rv_cxl_mbox_event(mbox_timeout);
  }

  assert(rv_cxl_mbox_cur == mbox_idle);
  return 0;
}
