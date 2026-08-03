// CXL PMEM passphrase flow, against the real state bits from
// drivers/cxl/cxlmem.h and the flag derivation in drivers/cxl/security.c.
//
// The device reports one state word; cxl_pmem_get_security_state() turns it
// into nvdimm flags. LOCKED and UNLOCKED are mutually exclusive, DISABLED
// means no passphrase is set, and FROZEN is orthogonal to all of them.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

extern unsigned int __VERIFIER_nondet_uint(void);

#define PASS_A "0123456789abcdef0123456789abcdef"
#define PASS_B "fedcba9876543210fedcba9876543210"

static void check_flag_consistency(u32 state)
{
  unsigned long u = cxl_pmem_security_flags(state, NVDIMM_USER);

  /* A device is never both locked and unlocked. */
  assert(!((u & NVDIMM_SECURITY_LOCKED) && (u & NVDIMM_SECURITY_UNLOCKED)));
  /* DISABLED means no user passphrase, and excludes the other two. */
  if (u & NVDIMM_SECURITY_DISABLED)
  {
    assert(!(state & CXL_PMEM_SEC_STATE_USER_PASS_SET));
    assert(!(u & (NVDIMM_SECURITY_LOCKED | NVDIMM_SECURITY_UNLOCKED)));
  }
  else
    assert(state & CXL_PMEM_SEC_STATE_USER_PASS_SET);

  /* The master view is derived independently of the user view. */
  unsigned long m = cxl_pmem_security_flags(state, NVDIMM_MASTER);
  assert(!((m & NVDIMM_SECURITY_DISABLED) && (m & NVDIMM_SECURITY_UNLOCKED)));
}

int main()
{
  struct cxl_pmem_security sec;
  u32 nd = __VERIFIER_nondet_uint();

  /* Whatever the device reports, the derivation stays self-consistent. */
  check_flag_consistency(nd);

  memset(&sec, 0, sizeof(sec));
  assert(cxl_pmem_security_flags(sec.state, NVDIMM_USER) ==
         NVDIMM_SECURITY_DISABLED);

  /* Set a user passphrase; the device then reports it as set and unlocked. */
  assert(cxl_pmem_set_passphrase(&sec, NVDIMM_USER, NULL, PASS_A) == 0);
  assert(sec.state & CXL_PMEM_SEC_STATE_USER_PASS_SET);
  assert(cxl_pmem_security_flags(sec.state, NVDIMM_USER) ==
         NVDIMM_SECURITY_UNLOCKED);

  /* Changing it needs the old one. */
  assert(cxl_pmem_set_passphrase(&sec, NVDIMM_USER, PASS_B, PASS_B) == -EACCES);
  assert(cxl_pmem_set_passphrase(&sec, NVDIMM_USER, PASS_A, PASS_B) == 0);

  /* A locked device unlocks only with the current passphrase. */
  sec.state |= CXL_PMEM_SEC_STATE_LOCKED;
  assert(cxl_pmem_security_flags(sec.state, NVDIMM_USER) &
         NVDIMM_SECURITY_LOCKED);
  assert(cxl_pmem_unlock(&sec, PASS_B) == 0);
  assert(!(sec.state & CXL_PMEM_SEC_STATE_LOCKED));

  /* Freeze first, then confirm the passphrase can no longer be changed. */
  assert(cxl_pmem_freeze(&sec) == 0);
  assert(cxl_pmem_security_flags(sec.state, NVDIMM_USER) &
         NVDIMM_SECURITY_FROZEN);
  assert(cxl_pmem_set_passphrase(&sec, NVDIMM_USER, PASS_B, PASS_A) == -EACCES);

  return 0;
}
