int nondet_int(void);
int main()
{
  /* x + INT_MIN must not become x - (int)2^31, i.e. x + 2^31. */
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x <= 10);
  int y = x + (-2147483647 - 1);
  int z = (-2147483647 - 1) + x;
  __ESBMC_assert(y < 0 && z < 0, "adding INT_MIN keeps its sign");
  return 0;
}
