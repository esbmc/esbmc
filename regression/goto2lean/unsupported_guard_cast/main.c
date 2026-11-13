int main(void) {
  unsigned x = 1;
  __ESBMC_assert(&x, "address as a guard");
  return 0;
}
