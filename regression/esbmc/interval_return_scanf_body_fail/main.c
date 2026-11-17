// Symex runs its scanf model whatever the body, so the return must not put
// back x's old value.
int scanf(const char *fmt, ...)
{
  return 1;
}
int main()
{
  int x = 0;
  scanf("%d", &x);
  __ESBMC_assert(x == 0, "x");
}
