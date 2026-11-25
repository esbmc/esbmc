int nondet_int(void);
int mid(int lo, int hi) { return lo + (hi - lo) / 2; }
int main(void) {
  int lo = nondet_int(), hi = nondet_int();
  __ESBMC_assume(lo <= hi);
  int m = mid(lo, hi);
  __ESBMC_assert(lo <= m && m <= hi, "midpoint in range");
  return 0;
}
