// CXL feature capability lookup (drivers/cxl/core/features.c).
//
// The supported-feature count comes back from Get Supported Features, so it
// describes the device, not the driver's table. Searching that table with a
// device-supplied count is what the clamp exists to prevent.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

int main()
{
  const u8 features[CXL_FEAT_MAX] = {
    CXL_FEAT_PATROL_SCRUB, CXL_FEAT_ECS, CXL_FEAT_SPPR, CXL_FEAT_HPPR};

  assert(cxl_feature_query(features, CXL_FEAT_MAX, CXL_FEAT_PATROL_SCRUB) == 0);
  assert(cxl_feature_query(features, CXL_FEAT_MAX, CXL_FEAT_HPPR) == 3);
  assert(cxl_feature_query(features, CXL_FEAT_MAX, 0x7F) == -ENOENT);

  /* A short count hides the tail of the table rather than reading past it. */
  assert(cxl_feature_query(features, 2, CXL_FEAT_SPPR) == -ENOENT);
  assert(cxl_feature_query(features, 0, CXL_FEAT_PATROL_SCRUB) == -ENOENT);

  /* However large the device claims, the search stays inside the table. */
  unsigned int nd = __VERIFIER_nondet_uint();
  int rc = cxl_feature_query(features, nd, CXL_FEAT_ECS);
  assert(rc == -ENOENT || (rc >= 0 && rc < CXL_FEAT_MAX));
  if (rc >= 0)
    assert(features[rc] == CXL_FEAT_ECS);

  return 0;
}
