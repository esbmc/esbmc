// CXL fabric manager that treats lockdown as advisory: it submits the hot-add
// and binds the endpoint regardless of what the policy gate returned.
// Expected: VERIFICATION FAILED (policy bypass: an endpoint is bound after
// lockdown, violating !cxl_hot_add_event / !switching_binding_event)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_fabric f;
  cxl_fabric_init(&f);

  cxl_fabric_lockdown(&f);
  unsigned int before = f.endpoints;

  /*
   * BUG: the gate's answer is discarded. -EPERM is not advice, and "the
   * device is physically there, so it must be enumerated" is exactly the
   * reasoning a lockdown exists to overrule. Having asked the policy is not
   * the same as having obeyed it.
   */
  cxl_fabric_submit(&f, CXL_FM_EV_HOT_ADD);

  /* Then the driver proceeds to bind it anyway, by hand. */
  f.state = CXL_FM_BIND_COMPLETE;
  f.endpoints++;

  /* The obligation: after lockdown the topology is fixed. */
  assert(f.endpoints == before);

  return 0;
}
