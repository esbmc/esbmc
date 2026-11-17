int __VERIFIER_nondet_int(void);
int main() {
  int a = __VERIFIER_nondet_int();
  int q = a / -2, r = a % -2;
  __ESBMC_assert(q * -2 + r == a, "identity");
  __ESBMC_assert(a >= 0 || r <= 0, "rem sign");
  __ESBMC_assert(q == -(a / 2), "neg divisor");
  return 0;
}
