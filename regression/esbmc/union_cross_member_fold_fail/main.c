/* Negative twin: the reinterpreted value is the WRITTEN one, sign
 * included — 0x8001 read as short is -32767, so the first assertion
 * holds only if the fold preserves the sign bit, and the second must
 * be refuted. */
typedef union { short s; unsigned short r; } slot_t;
static slot_t g;
int main(void)
{
  g.r = 0x8001;
  __ESBMC_assert(g.s == -32767, "sign carried through the reinterpretation");
  __ESBMC_assert(g.s == 7, "any other value must be refuted");
  return 0;
}
