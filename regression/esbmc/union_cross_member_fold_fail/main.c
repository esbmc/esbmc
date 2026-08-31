/* Negative twin: the reinterpreted value is the WRITTEN one — asserting
 * any other concrete value must be refuted. */
typedef union { short s; unsigned short r; } slot_t;
static slot_t g;
int main(void)
{
  g.r = 0x8001;
  __ESBMC_assert(g.s == 7, "reinterpretation must carry the real bits");
  return 0;
}
