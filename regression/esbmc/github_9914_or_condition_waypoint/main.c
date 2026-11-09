// An operand of || gets no branching waypoint: its own direction is not the
// condition's. A lone condition keeps its waypoint.
int nondet_int();

int main()
{
  int a = nondet_int(), b = nondet_int();
  if (a > 0 || nondet_int())
    return 0;
  if (b > 3)
    b = 0;
  __ESBMC_assert(b != 0, "reached");
  return 0;
}
