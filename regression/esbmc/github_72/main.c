// #72: a write through a pointer that may not point anywhere valid must not
// emit an SSA assignment to the failed-dereference symbol.
int nondet_int(void);
int x;
int main(void)
{
  int *p;
  if (nondet_int())
    p = &x;
  *p = 1;
  __ESBMC_assert(x == 0 || x == 1, "x keeps a defined value");
  return 0;
}
