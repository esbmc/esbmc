unsigned nondet_uint(void);
int main(void) {
  unsigned n = nondet_uint();
  unsigned s = 0, i = 0;
  while (i < n) {
    s = s + 1;
    i = i + 1;
  }
  __ESBMC_assert(s == n, "s counts the iterations");
  return 0;
}
