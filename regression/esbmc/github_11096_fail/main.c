// Issue #11096: havoc of a pointer with an unknown target was not checked.
void __ESBMC_havoc_object(void *);
extern char *f(void);
int main(void)
{
  char *q = f();
  __ESBMC_assume(q == "lit");
  __ESBMC_havoc_object(q);
  return 0;
}
