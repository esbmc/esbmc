unsigned __VERIFIER_nondet_uint(void);
int main() {
  unsigned x = __VERIFIER_nondet_uint();
  __ESBMC_assume((int)x == -3);
  unsigned q = x / 2u, r = x % 2u;
  __ESBMC_assert((int)q == -1, "tdiv q");
  __ESBMC_assert((int)r == -1, "tmod r");
  return 0;
}
