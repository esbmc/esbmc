// CXL PMEM driver that re-keys a device it has already frozen.
// Expected: VERIFICATION FAILED (the RV monitor rejects the sequence)

#include <assert.h>
#include "rv_cxl_sec.h"

int main(void)
{
  rv_cxl_sec_reset();

  rv_cxl_sec_event(set_passphrase);
  rv_cxl_sec_event(freeze);

  /*
   * BUG: freeze is irreversible for the life of the device -- CXL 2.0
   * §8.2.9.8.6.3 has Set Passphrase refused once security is frozen. A driver
   * that treats freeze as advisory, or that reuses a rekey path without
   * re-reading the security state, issues this and gets a hardware error it
   * has no handler for.
   */
  rv_cxl_sec_event(set_passphrase);

  return 0;
}
