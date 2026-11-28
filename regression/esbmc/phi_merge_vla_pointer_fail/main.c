int nondet_int();

int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 2 && n <= 4);
  int a[n];
  int b[n];
  int(*p)[n] = &a;
  while (nondet_int())
  {
    if (nondet_int())
      p = &b;
    if (nondet_int())
      break;
  }
  (*p)[1] = 7;
  __ESBMC_assert((*p)[1] == 8, "store through merged VLA pointer");
  return 0;
}
