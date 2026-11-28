// With nondet lanes, vmaxq_s16 and vminq_s16 bound each lane, and vgetq_lane
// reads back the lane vsetq_lane wrote.
#include <arm_neon.h>
#include <assert.h>

short nondet_short(void);

int main(void)
{
  int16x8_t a, b;
  for (int i = 0; i < 8; i++)
  {
    a[i] = nondet_short();
    b[i] = nondet_short();
  }
  int16x8_t mx = vmaxq_s16(a, b), mn = vminq_s16(a, b);
  assert(mx[5] >= a[5] && mx[5] >= b[5] && (mx[5] == a[5] || mx[5] == b[5]));
  assert(mn[2] <= a[2] && mn[2] <= b[2] && (mn[2] == a[2] || mn[2] == b[2]));
  short x = nondet_short();
  assert(vgetq_lane_s16(vsetq_lane_s16(x, a, 6), 6) == x);
  int16x8_t u = vuzp1q_s16(a, b);
  assert(u[1] == a[2] && u[7] == b[6]);
  return 0;
}
