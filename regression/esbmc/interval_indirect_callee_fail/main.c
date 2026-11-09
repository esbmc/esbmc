void g(int a)
{
  __ESBMC_assert(a != 5, "a may be 5");
}
void h(int x)
{
  g(x);
}
int main()
{
  void (*fp)(int) = h;
  g(2);
  fp(5);
  return 0;
}
