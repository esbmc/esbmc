int main() {
  float expected = nondet_float();
  float actual = __builtin_nontemporal_load(&expected);
  __ESBMC_assert(expected == actual, "check");
  return 0;
}
