/* Width-guard for the same-width cross-member fold: a DIFFERING-width
 * cross-member read is not a bit reinterpretation and must keep byte
 * semantics. After c = -1 the little-endian bytes are ff 00 00 00, so
 * i reads 255; a relaxation of the width check would typecast-fold the
 * stored -1 instead and refute this. */
typedef union
{
  signed char c;
  int i;
} u_t;
static u_t u;
int main(void)
{
  u.c = -1;
  __ESBMC_assert(u.i == 255, "cross-width read keeps byte semantics");
  return 0;
}
