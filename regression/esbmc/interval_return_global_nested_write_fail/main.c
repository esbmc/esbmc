// f() writes g only through h(), so the return from f() must not put back g.
int g;
void h(void)
{
  g = 5;
}
void f(void)
{
  h();
}
int main()
{
  g = 0;
  f();
  __ESBMC_assert(g == 0, "g");
}
