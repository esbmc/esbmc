/* #9000: a pointer that may address a local whose lifetime has ended does
 * not satisfy is_fresh. */
_Bool nondet_bool();
int y;
void g(int *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(int)));
  __ESBMC_ensures(1);
  __ESBMC_assigns();
}
int main()
{
  int *p;
  if (nondet_bool())
  {
    int x;
    p = &x;
  }
  else
    p = &y;
  g(p);
  return 0;
}
