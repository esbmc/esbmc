int main(void) {
  _Bool x = 0;
  int n = 0;
  while (1) { __ESBMC_assert(!x, "P"); x = (n == 3); n++; }
}
