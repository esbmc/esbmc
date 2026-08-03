// CXL EDAC patrol scrub cycle programming (drivers/cxl/core/edac.c).
//
// The scrub control register packs the current cycle in the low byte and the
// device's advertised minimum in the high byte. A cycle shorter than that
// minimum is a property of the media, not a suggestion.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned short __VERIFIER_nondet_ushort(void);
extern unsigned char __VERIFIER_nondet_uchar(void);

int main()
{
  u16 out;

  /* Minimum 8 hours, currently 24. */
  u16 reg = (u16)((8U << 8) | 24U);

  assert(cxl_edac_set_patrol_scrub(reg, 12, &out) == 0);
  assert((out & CXL_SCRUB_CONTROL_CYCLE_MASK) == 12);
  /* Programming the cycle must not disturb the advertised minimum. */
  assert((out & CXL_SCRUB_CONTROL_MIN_CYCLE_MASK) ==
         (reg & CXL_SCRUB_CONTROL_MIN_CYCLE_MASK));

  assert(cxl_edac_set_patrol_scrub(reg, 8, &out) == 0);
  assert(cxl_edac_set_patrol_scrub(reg, 7, &out) == -EINVAL);
  assert(cxl_edac_set_patrol_scrub(reg, 0, &out) == -EINVAL);

  /* Whatever the device advertises, acceptance implies the request met it,
     and the readback round-trips. */
  u16 nd_reg = __VERIFIER_nondet_ushort();
  u8 nd_hours = __VERIFIER_nondet_uchar();
  if (cxl_edac_set_patrol_scrub(nd_reg, nd_hours, &out) == 0)
  {
    u8 min = (u8)((nd_reg & CXL_SCRUB_CONTROL_MIN_CYCLE_MASK) >> 8);
    assert(nd_hours >= min);
    assert((out & CXL_SCRUB_CONTROL_CYCLE_MASK) == nd_hours);
    assert(((out & CXL_SCRUB_CONTROL_MIN_CYCLE_MASK) >> 8) == min);
  }

  return 0;
}
