void h(int a)
{
  __ESBMC_assert(a != 5, "a is never 5");
}
int main()
{
  void (*fp)(int) = h;
  h(2);
  fp(3);
  return 0;
}
