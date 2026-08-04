// Linux Runtime Verifier monitor 'snep', checked statically by ESBMC.
//
// snep ("schedule not entering preemptive") is a real kernel RV monitor,
// kernel/trace/rv/monitors/snep/. rv_snep.h is generated from its rvgen
// output by scripts/rv2c.py; the automaton and its rejected transitions are
// the kernel's, not invented here.
//
// The obligation is the one in include/rv/da_monitor.h:690 -- an event with
// no transition from the current state drives the automaton to INVALID_STATE.
// That is G (state != INVALID_STATE): safety, so an assertion discharges it.
//
// What this adds over RV itself: RV observes the interleavings that actually
// happened on a running kernel. Here the event producer is symbolic, so every
// sequence it could emit is covered at once.
// Expected: VERIFICATION SUCCESSFUL

#include <assert.h>
#include "rv_snep.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

/* A producer that respects the protocol: preemption may be toggled freely
   outside scheduling context, and a schedule_entry is always matched by a
   schedule_exit before anything else happens. */
int main(void)
{
  rv_snep_reset();

  for (int i = 0; i < 4; i++)
  {
    unsigned int c = __VERIFIER_nondet_uint();
    __ESBMC_assume(c < 3);

    if (c == 0)
      rv_snep_event(preempt_disable_snep);
    else if (c == 1)
      rv_snep_event(preempt_enable_snep);
    else
    {
      /* Entering scheduling context commits us to leaving it before doing
         anything else -- which is exactly what the automaton encodes. */
      rv_snep_event(schedule_entry_snep);
      rv_snep_event(schedule_exit_snep);
    }
  }

  /* Back where we started, having never been rejected. */
  assert(rv_snep_cur == non_scheduling_context_snep);
  return 0;
}
