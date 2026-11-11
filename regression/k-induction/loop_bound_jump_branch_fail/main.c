int nondet_int();
unsigned nondet_uint();

int main()
{
  unsigned x = nondet_uint();
  // No loop reads x, so 100000 must not become the base case's bound.
  if (x > 100000)
    return 0;
  unsigned c = 0, d = 0;
  while (nondet_int())
  {
    c++;
    while (nondet_int())
      d++;
  }
  __ESBMC_assert(c != 4, "shallow");
}
