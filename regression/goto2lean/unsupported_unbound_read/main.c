unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint();
  if (c) goto L;
  unsigned y = 1;
L:
  y = y + 1;
  __ESBMC_assert(y != 0, "y");
  return 0;
}
