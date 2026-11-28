int main() {
  unsigned x0 = 0, x1 = 0, x2 = 0, n = 0;
#pragma unroll 4
  for (;;) {
    x2 = x1; x1 = x0; x0 = (n == 6);
    n++;
    __ESBMC_assert(x2 == 0, "x2 == 0");
  }
}
