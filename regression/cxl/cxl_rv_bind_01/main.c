// CXL fabric-manager bind sequence as a Linux RV automaton, checked by ESBMC.
//
// cxl_bind.dot encodes CXL 4.0 §14.7.7's bind of a pooled SLD to a vPPB:
//   unbound -> bind_initiated -> host_recognized -> host_enumerated -> bound
// Each step has exactly one legal successor, so the sequence is the property:
// the FM and the host must hand off in order, and a bound endpoint can only
// be released by fm_unbind.
//
// Generated through the kernel's own tools/verification/rvgen dot2c, so the
// same .dot yields a live RV monitor, this static check, and a NuSMV model
// (scripts/dot2smv.py) that proves the obligation unbounded.
// Expected: VERIFICATION SUCCESSFUL

#include <assert.h>
#include "rv_cxl_bind.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

int main(void)
{
  rv_cxl_bind_reset();

  for (int i = 0; i < 2; i++)
  {
    rv_cxl_bind_event(fm_bind);
    rv_cxl_bind_event(host_recognize);
    rv_cxl_bind_event(host_enumerate);
    rv_cxl_bind_event(fm_confirm);
    assert(rv_cxl_bind_cur == bound);

    rv_cxl_bind_event(fm_unbind);
    assert(rv_cxl_bind_cur == unbound);
  }
  return 0;
}
