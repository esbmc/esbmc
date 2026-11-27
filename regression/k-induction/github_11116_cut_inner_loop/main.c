int nondet_int();
int main() {
  int x = 0;
  for (;;) {
    int n = nondet_int();
    __ESBMC_assume(n >= 5 && n <= 6);
    for (int i = 0; i < n; i++)
      ;
    __ESBMC_assert(x != 0, "x != 0");
    x = 1;
  }
}
