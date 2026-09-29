/* The bracket, with the products computed EXACTLY on raw integers instead of
 * in u0.8. r*r in the fixed-point type rounds to 8 fractional bits, which
 * makes it useless as a bracket -- the comparison has to be done at full
 * width:
 *
 *   r*r <= x < (r+1)*(r+1)   as integers, where x is scaled by 2^8
 *
 * i.e. raw_r^2 <= raw_x * 256 < (raw_r+1)^2. That is the exact statement of
 * "square root of x/256 truncated to 8 fractional bits". */
unsigned short _Fract __ESBMC_fxp_sqrt_uhr(unsigned short _Fract);
unsigned char nondet_uchar(void);

int main(void)
{
  unsigned char xb = nondet_uchar();
  unsigned short _Fract x;
  __ESBMC_bitcast(&x, &xb);

  unsigned short _Fract r = __ESBMC_fxp_sqrt_uhr(x);
  unsigned char rb;
  __ESBMC_bitcast(&rb, &r);

  /* exact integer arithmetic, no fixed-point rounding involved */
  unsigned int rr = (unsigned int)rb * (unsigned int)rb;
  unsigned int nn = ((unsigned int)rb + 1) * ((unsigned int)rb + 1);
  unsigned int xs = (unsigned int)xb * 256u;

  __ESBMC_assert(rr <= xs, "raw_r^2 <= raw_x * 2^8");
  __ESBMC_assert(nn > xs, "(raw_r+1)^2 > raw_x * 2^8");
  return 0;
}
