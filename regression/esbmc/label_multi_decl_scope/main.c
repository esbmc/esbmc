int main()
{
L:
  int a = 1, b = 2;
  int *p = &b;
  __ESBMC_assert(*p == 2, "b is still alive after its labelled declaration");
  return a;
}
