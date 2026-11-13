unsigned nondet_uint(void);
/* A return inside the loop leaves through the loop function's `ret`. */
unsigned find(unsigned n, unsigned key) {
  for (unsigned i = 0; i < n; i++)
    if (i == key)
      return i;
  return n;
}
int main(void) {
  unsigned n = nondet_uint(), key = nondet_uint();
  __ESBMC_assume(n <= 8);
  unsigned r = find(n, key);
  __ESBMC_assert(r <= n, "result in range");
  return 0;
}
