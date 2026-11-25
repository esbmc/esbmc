// #72: with the write to the failed-dereference sink dropped, the write
// through the valid target is still seen.
int nondet_int(void);
int x;
int main(void)
{
  int *p;
  if (nondet_int())
    p = &x;
  *p = 1;
  __ESBMC_assert(x == 0, "x is never written");
  return 0;
}
