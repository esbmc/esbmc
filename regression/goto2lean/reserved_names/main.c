unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint();
  unsigned result = c, post = c, ret = c;
  __ESBMC_assert(result == post && post == ret, "same");
  return 0;
}
