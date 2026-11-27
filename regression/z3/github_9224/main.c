// #9224: signed / and % by 2^k truncate toward zero (C11 6.5.5p6). The
// expected values are built from shifts and masks of the magnitude.
int nondet_int(void);
long long nondet_ll(void);
int main(void)
{
  int x = nondet_int();
  long long mx = x < 0 ? -(long long)x : x;
  int q = (int)(mx >> 2), r = (int)(mx & 7);
  __ESBMC_assert(x / 4 == (x < 0 ? -q : q), "x / 4");
  __ESBMC_assert(x % 8 == (x < 0 ? -r : r), "x % 8");
  __ESBMC_assert(x / 1073741824 * 1073741824 + x % 1073741824 == x, "2^30");

  long long y = nondet_ll();
  unsigned long long my = y < 0 ? -(unsigned long long)y : y;
  long long qy = (long long)(my >> 1), ry = (long long)(my & 15);
  __ESBMC_assert(y / 2 == (y < 0 ? -qy : qy), "y / 2");
  __ESBMC_assert(y % 16 == (y < 0 ? -ry : ry), "y % 16");
  return 0;
}
