unsigned int nondet_uint();

int main()
{
  unsigned int n = nondet_uint();
  unsigned int x = n, y = 0;
  __ESBMC_loop_invariant(x + y == n);
  while (x > 0)
  {
    x--;
    y++;
  }
  assert(y == n);
}
