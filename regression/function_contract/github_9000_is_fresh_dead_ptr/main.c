/* #9000: a pointer that may address a live local or a global satisfies
 * is_fresh. */
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
  int x;
  if (nondet_bool())
    p = &x;
  else
    p = &y;
  g(p);
  return 0;
}
