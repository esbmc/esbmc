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
  __ESBMC_assert(
    __ESBMC_forall(&x, h() == 12), "reordered evaluation would prove this");
}
