// CXL driver that commits a region before attaching any target to it.
// Expected: VERIFICATION FAILED (the RV monitor rejects the sequence)

#include <assert.h>
#include "rv_cxl_region.h"

int main(void)
{
  rv_cxl_region_reset();

  rv_cxl_region_event(set_size);

  /*
   * BUG: a region with its size set but no targets attached is not
   * configured. Committing here reaches CXL_CONFIG_COMMIT with nothing behind
   * it, and cxl_dax_region_alloc() will then happily build a DAX device over
   * an interleave that decodes to no device.
   */
  rv_cxl_region_event(commit);

  return 0;
}
