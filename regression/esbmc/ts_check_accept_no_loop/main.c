int nondet_int();

int main() {
  int x = nondet_int();
  __ESBMC_assert(x != 5, "x is never five");
  return 0;
}
