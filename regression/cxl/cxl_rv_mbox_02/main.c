// CXL mailbox driver that submits a second command without waiting for the
// first, checked against the same Linux RV automaton.
// Expected: VERIFICATION FAILED (the RV monitor rejects the sequence)

#include <assert.h>
#include "rv_cxl_mbox.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

int main(void)
{
  rv_cxl_mbox_reset();

  rv_cxl_mbox_event(mbox_submit);

  /*
   * BUG: step 1 of CXL 2.0 §8.2.8.4 is "read MB Control to verify the doorbell
   * is clear". This driver skips it and submits again while the first command
   * is still in flight -- overwriting the command register under a device
   * that is still reading it. On hardware the symptom is a lost or corrupted
   * command, not a crash, which is why it survives testing.
   */
  if (__VERIFIER_nondet_uint() & 1u)
    rv_cxl_mbox_event(doorbell_clear);

  rv_cxl_mbox_event(mbox_submit);

  return 0;
}
