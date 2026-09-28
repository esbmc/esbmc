/* Arrays, structs and floats must all reach the solver flattened to pure
 * QF_BV: the stand-in runs a real z3 on the emitted formula, so any leaked
 * Array or FloatingPoint sort fails the (set-logic QF_BV) run loudly. */
struct point
{
  int x;
  float y;
};

int main()
{
  int a[4];
  struct point p;
  unsigned i = nondet_uint() % 4;
  a[i] = 42;
  p.x = a[i];
  p.y = 1.5f;
  __ESBMC_assert(p.x == 41 && p.y == 1.5f, "flattened features round-trip (unsafe)");
  return 0;
}
