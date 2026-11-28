int main() {
  unsigned x0 = 0, x1 = 0, x2 = 0;
  for (;;) {
    x2 = x1; x1 = x0; x0 = 0;
    for (int i = 0; i < 2; i++)
      __ESBMC_assert(x2 == 0, "x2 == 0");
  }
}
