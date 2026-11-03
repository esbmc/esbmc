int nondet_int(void);
int calls;

void fail(void)
{
  __ESBMC_assert(0, "callee assertion");
  calls++;
}

int main()
{
  int a = nondet_int();
  int b = nondet_int();
  fail();
  if (a > 0 && b > 0)
    return 2;
  if (a < 0 || b < 0)
    return 1;
  return 0;
}
