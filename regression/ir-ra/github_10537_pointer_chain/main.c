int nondet_int(void);
int main()
{
  /* The two int offsets of this pointer chain sum past INT_MAX: folding them
   * into one int constant would wrap the offset to -2. */
  char buf[4];
  char *q = buf;
  int i = nondet_int();
  char *r = q + 2147483647 + i + 2147483647;
  __ESBMC_assert(r - q == (long)i + 4294967294L, "offsets are not wrapped");
  return 0;
}
