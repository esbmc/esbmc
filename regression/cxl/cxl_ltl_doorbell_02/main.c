// CXL mailbox doorbell wait as an LTL response property, WITHOUT the timeout.
//
// BUG: cxl_pci_mbox_wait_for_doorbell() gives up after
// CXL_MAILBOX_TIMEOUT_MS. This variant polls and then simply proceeds, so a
// device that never clears the doorbell leaves the command outstanding for
// ever and the response obligation
//
//     G (doorbell_busy -> F mbox_settled)
//
// is never discharged.
//
// ESBMC nonetheless reports the same outcome as cxl_ltl_doorbell_01, which
// has the timeout: LTL_FAILING for both. `G (p -> F q)` is pure liveness and
// has no finite counterexample, so the automaton's _ltl2ba_bad_prefix_states
// is all-false and LTL_BAD is unreachable whatever the program does. This
// test is therefore NOT evidence that the bug is detected -- it is evidence
// that this property class is out of reach for bounded checking, which is
// worth having in the suite rather than in someone's memory.
// Expected: VERIFICATION SUCCESSFUL, LTL_FAILING (indistinguishable from _01)

int doorbell_busy = 0;
int mbox_settled = 1; /* nothing outstanding before the first submit */

#define CXL_MAILBOX_POLL_LIMIT 4

int main()
{
  for (int cmd = 0; cmd < 2; cmd++)
  {
    /* Submit: set the doorbell, and the command is now outstanding. */
    doorbell_busy = 1;
    mbox_settled = 0;

    for (int i = 0; i < CXL_MAILBOX_POLL_LIMIT; i++)
    {
      if (nondet_int())
      {
        /* Device cleared the doorbell. */
        doorbell_busy = 0;
        mbox_settled = 1;
        break;
      }
    }

    /* No timeout: if the device never cleared the doorbell, the command
       stays outstanding and the driver keeps waiting. */
  }

  return 0;
}
