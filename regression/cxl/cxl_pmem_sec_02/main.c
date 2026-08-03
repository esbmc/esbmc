// CXL PMEM driver that unlocks with a passphrase it has not confirmed, and
// treats the attempt as having succeeded.
// Expected: VERIFICATION FAILED (driver bug: unlock result dropped, device
// left locked while treated as unlocked)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

#define PASS_GOOD "0123456789abcdef0123456789abcdef"
#define PASS_WRONG "fedcba9876543210fedcba9876543210"

int main()
{
  struct cxl_pmem_security sec;

  memset(&sec, 0, sizeof(sec));
  if (cxl_pmem_set_passphrase(&sec, NVDIMM_USER, NULL, PASS_GOOD))
    return 0;
  sec.state |= CXL_PMEM_SEC_STATE_LOCKED;

  /*
   * BUG: the unlock result is dropped. A wrong passphrase leaves the device
   * locked and burns an attempt against the limit -- CXL 2.0 §8.2.9.8.6.4 --
   * so proceeding as though the media were readable is wrong twice over.
   * Having sent Unlock is not the same as being unlocked.
   */
  cxl_pmem_unlock(&sec, PASS_WRONG);

  assert(!(cxl_pmem_security_flags(sec.state, NVDIMM_USER) &
           NVDIMM_SECURITY_LOCKED));

  return 0;
}
