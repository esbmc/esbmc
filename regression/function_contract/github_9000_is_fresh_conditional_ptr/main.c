/* #9000: a pointer that may be &s or NULL, then assumed non-NULL, still
 * satisfies is_fresh for the size of s. */
_Bool nondet_bool();
struct P
{
  int a;
};
void g(struct P *d)
{
  __ESBMC_requires(__ESBMC_is_fresh(d, sizeof(struct P)));
  __ESBMC_ensures(1);
  d->a = 1;
}
int main()
{
  struct P s;
  struct P *d = nondet_bool() ? &s : 0;
  __ESBMC_assume(d != 0);
  g(d);
  return 0;
}
