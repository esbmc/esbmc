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
// Expected: VERIFICATION FAILED (the producer emits a sequence the kernel
// monitor rejects)

#include <assert.h>
#include "rv_snep.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

/*
 * BUG: this producer toggles preemption without regard to scheduling
 * context. snep exists precisely because that is not allowed: from
 * scheduling_contex every event except schedule_exit is rejected.
 */
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
      rv_snep_event(schedule_entry_snep);  /* and then just carries on */
  }

  return 0;
}
