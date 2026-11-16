unsigned nondet_uint(void);
int main(void) {
  unsigned a = nondet_uint(), b = nondet_uint(), x = 0;
  if (a) {
    if (b) {
      x = 1;
      goto out;
    }
    x = 2;
  }
  x = 3;
out:
  __ESBMC_assert(x != 0, "x");
  return 0;
}
