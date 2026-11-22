unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint();
  if (x == 3) __ESBMC_unreachable();
  return 0;
}
