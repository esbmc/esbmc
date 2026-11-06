int nondet_int();
int g;

void cs(void)
{
  if (nondet_int())
    g++;
}

void t2(void)
{
  cs();
}

void t1(void)
{
  cs();
  if (nondet_int())
    g += 10;
}

int main()
{
  void (*f)(void) = nondet_int() ? t1 : t2;
  f();
  __ESBMC_assert(g != 11, "g");
  return 0;
}
