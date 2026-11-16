/* #9000: a pointer that may be &s or NULL does not satisfy is_fresh. */
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
  g(d);
  return 0;
}
