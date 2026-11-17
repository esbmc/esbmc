// f() may call w() through fp, so the return must not put back g.
int g;
void w(void)
{
  g = 5;
}
void r(void)
{
}
void (*fp)(void);
void f(void)
{
  fp();
}
int main()
{
  fp = nondet_int() ? w : r;
  g = 0;
  f();
  __ESBMC_assert(g == 0, "g");
}
