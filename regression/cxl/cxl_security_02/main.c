// CXL invalid security state transition test.
// Tests that the driver rejects invalid transitions.
// Expected: VERIFICATION FAILED (driver bug: allows invalid transition)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

enum cxl_security_state {
  CXL_SEC_NONE = 0,
  CXL_SEC_UNLOCKED,
  CXL_SEC_LOCKED,
  CXL_SEC_DISABLED,
  CXL_SEC_PASSPHRASE_SET,
};

struct cxl_dev {
  enum cxl_security_state state;
};

static struct cxl_dev test_cxld;

/*
 * BUG: This driver allows ALL transitions, including invalid ones.
 * A correct implementation should only allow spec-defined transitions.
 */
int cxl_set_security(struct cxl_dev *cxld, enum cxl_security_state new_state)
{
  (void)new_state;
  /* BUG: No validation — accepts any transition */
  cxld->state = new_state;
  return 0;
}

int main()
{
  test_cxld.state = CXL_SEC_NONE;

  /*
   * BUG: NONE -> UNLOCKED is NOT a valid CXL transition.
   * Must go through PASSPHRASE_SET first.
   * The driver should reject this but doesn't.
   */
  int ret = cxl_set_security(&test_cxld, CXL_SEC_UNLOCKED);
  assert(ret == 0); /* Bug: returns success for invalid transition */

  /*
   * Verify the invariant: from NONE, only PASSPHRASE_SET or DISABLED
   * should be allowed. UNLOCKED is invalid.
   */
  __ESBMC_assert(test_cxld.state != CXL_SEC_UNLOCKED ||
                 test_cxld.state == CXL_SEC_PASSPHRASE_SET ||
                 test_cxld.state == CXL_SEC_DISABLED,
                 "NONE cannot transition directly to UNLOCKED");
}
