unsigned mid(unsigned lo, unsigned hi) {
  __ESBMC_assume(lo <= hi);
  unsigned m = (lo + hi) / 2;
  __ESBMC_assert(lo <= m, "lo<=m");
  return m;
}
int main(void) { return 0; }
