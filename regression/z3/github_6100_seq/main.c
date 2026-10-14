int g;

int bump(void)
{
  return ++g;
}

int h(void)
{
  int a = bump();
  return bump() * 10 + a;
}

int main()
{
  int x;
  __ESBMC_assert(__ESBMC_forall(&x, h() == 21), "h returns 21 on first call");
}
