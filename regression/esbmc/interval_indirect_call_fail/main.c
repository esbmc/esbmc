void h(int a)
{
  __ESBMC_assert(a != 5, "a may be 5");
}
int main()
{
  void (*fp)(int) = h;
  h(2);
  fp(5);
  return 0;
}
