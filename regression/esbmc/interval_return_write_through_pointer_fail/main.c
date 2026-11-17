// f() writes g through p, so the return must not put back g's old value.
int g;
int *p = &g;
void f(void)
{
  *p = 5;
}
int main()
{
  g = 0;
  f();
  __ESBMC_assert(g == 0, "g");
}
