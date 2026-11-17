int nondet_int(void);
int clamp(int x, int lo, int hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}
int sign(int x) {
  int s;
  if (x > 0)
    s = 1;
  else if (x < 0)
    s = -1;
  else
    s = 0;
  return s;
}
int main(void) {
  int x = nondet_int(), lo = nondet_int(), hi = nondet_int();
  __ESBMC_assume(lo <= hi);
  int c = clamp(x, lo, hi);
  __ESBMC_assert(lo <= c && c <= hi, "clamp in range");
  __ESBMC_assert(sign(c - c) == 0, "sign of zero");
  return 0;
}
