// Issue #11096: havoc of a pointer to a known object is still accepted.
void __ESBMC_havoc_object(void *);
int main(void)
{
  char buf[4] = {0};
  char *q = buf;
  __ESBMC_havoc_object(q);
  return 0;
}
