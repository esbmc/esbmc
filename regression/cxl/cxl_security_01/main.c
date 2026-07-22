// CXL security state transition test.
// Tests that security state transitions follow the CXL spec.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* CXL security states per CXL 2.0 spec */
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

int cxl_set_security(struct cxl_dev *cxld, enum cxl_security_state new_state)
{
  assert(cxld != NULL);

  /* Valid transitions per CXL 2.0 spec */
  switch (cxld->state)
  {
  case CXL_SEC_NONE:
    /* NONE -> PASSPHRASE_SET or DISABLED */
    if (new_state == CXL_SEC_PASSPHRASE_SET || new_state == CXL_SEC_DISABLED)
    {
      cxld->state = new_state;
      return 0;
    }
    break;

  case CXL_SEC_PASSPHRASE_SET:
    /* PASSPHRASE_SET -> UNLOCKED (after unlock) or DISABLED */
    if (new_state == CXL_SEC_UNLOCKED || new_state == CXL_SEC_DISABLED)
    {
      cxld->state = new_state;
      return 0;
    }
    break;

  case CXL_SEC_UNLOCKED:
    /* UNLOCKED -> LOCKED or PASSPHRASE_SET */
    if (new_state == CXL_SEC_LOCKED || new_state == CXL_SEC_PASSPHRASE_SET)
    {
      cxld->state = new_state;
      return 0;
    }
    break;

  case CXL_SEC_LOCKED:
    /* LOCKED -> PASSPHRASE_SET or DISABLED */
    if (new_state == CXL_SEC_PASSPHRASE_SET || new_state == CXL_SEC_DISABLED)
    {
      cxld->state = new_state;
      return 0;
    }
    break;

  case CXL_SEC_DISABLED:
    /* DISABLED -> NONE */
    if (new_state == CXL_SEC_NONE)
    {
      cxld->state = new_state;
      return 0;
    }
    break;
  }

  return -1; /* Invalid transition */
}

int main()
{
  test_cxld.state = CXL_SEC_NONE;

  /* NONE -> PASSPHRASE_SET */
  assert(cxl_set_security(&test_cxld, CXL_SEC_PASSPHRASE_SET) == 0);
  assert(test_cxld.state == CXL_SEC_PASSPHRASE_SET);

  /* PASSPHRASE_SET -> UNLOCKED */
  assert(cxl_set_security(&test_cxld, CXL_SEC_UNLOCKED) == 0);
  assert(test_cxld.state == CXL_SEC_UNLOCKED);

  /* UNLOCKED -> LOCKED */
  assert(cxl_set_security(&test_cxld, CXL_SEC_LOCKED) == 0);
  assert(test_cxld.state == CXL_SEC_LOCKED);

  /* LOCKED -> DISABLED */
  assert(cxl_set_security(&test_cxld, CXL_SEC_DISABLED) == 0);
  assert(test_cxld.state == CXL_SEC_DISABLED);

  /* DISABLED -> NONE (cold reset) */
  assert(cxl_set_security(&test_cxld, CXL_SEC_NONE) == 0);
  assert(test_cxld.state == CXL_SEC_NONE);
}
