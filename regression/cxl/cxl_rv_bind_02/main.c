// CXL fabric manager that confirms the binding before the host has enumerated
// the endpoint.
// Expected: VERIFICATION FAILED (the RV monitor rejects the sequence)

#include <assert.h>
#include "rv_cxl_bind.h"

int main(void)
{
  rv_cxl_bind_reset();

  rv_cxl_bind_event(fm_bind);
  rv_cxl_bind_event(host_recognize);

  /*
   * BUG: §14.7.7 has the host enumerate the SLD before the FM indicates the
   * binding is complete. Skipping the enumeration is invisible on a device
   * that happens to enumerate quickly -- the FM confirms a topology the host
   * has not yet built, and the race only shows under load.
   */
  rv_cxl_bind_event(fm_confirm);

  return 0;
}
