int main(void) {
  _Bool x = 0;
  while (1) { __ESBMC_assert(!x, "P"); x = 0; }
}
