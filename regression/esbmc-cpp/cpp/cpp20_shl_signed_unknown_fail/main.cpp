// Unconstrained signed int may be negative; the type-driven non-negativity
// predicate falls through and the overflow claim is preserved.

extern "C" int nondet_int();

int main()
{
  int x = nondet_int();
  int y = x << 1;
  return y;
}
