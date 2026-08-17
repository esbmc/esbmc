// CXL fabric lockdown: once the fabric enters runtime, no endpoint may be
// hot-added, bound, unbound or reconfigured.
//
// Realises three safety obligations from an external intent decomposition
// (Seccom), each derived from a CXL 4.0 section:
//   !cxl_hot_add_event        §14.7.5
//   !switching_binding_event  §14.7.7
//   !runtime_config_trigger   §14.7
//
// All three are invariants over the event stream, not liveness properties --
// `G (!event)` needs no Buchi automaton, only an assertion at every point the
// event could occur. See docs/cxl-temporal-properties.md.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

int main()
{
  struct cxl_fabric f;
  cxl_fabric_init(&f);

  /* Before lockdown the fabric composes normally: a bind runs the §14.7.7
     sequence to completion. */
  assert(cxl_fabric_submit(&f, CXL_FM_EV_BIND) == 0);
  assert(cxl_fabric_bind_completed(&f));
  assert(f.endpoints == 1);

  cxl_fabric_lockdown(&f);

  unsigned int before = f.endpoints;

  /* Whatever event sequence is attempted afterwards, every one of the three
     denied classes is refused, and none of them changes the topology. */
  for (int i = 0; i < 3; i++)
  {
    unsigned int e = __VERIFIER_nondet_uint();
    __ESBMC_assume(e >= CXL_FM_EV_HOT_ADD && e <= CXL_FM_EV_RUNTIME_CONFIG);

    int rc = cxl_fabric_submit(&f, (enum cxl_fm_event)e);

    assert(rc == -EPERM);
    assert(f.endpoints == before);
  }

  /* The gate counted every refusal, and lockdown never lifts. */
  assert(f.denied == 3);
  assert(f.locked);

  return 0;
}
