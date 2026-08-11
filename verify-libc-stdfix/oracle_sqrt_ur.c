unsigned _Fract __ESBMC_fxp_sqrt_ur(unsigned _Fract);
unsigned short nondet_ushort(void);
int main(void){
  unsigned short xb = nondet_ushort();
  unsigned _Fract x; __ESBMC_bitcast(&x, &xb);
  unsigned _Fract r = __ESBMC_fxp_sqrt_ur(x);
  unsigned short rb; __ESBMC_bitcast(&rb, &r);
  unsigned long long rr=(unsigned long long)rb*rb;
  unsigned long long nn=((unsigned long long)rb+1)*((unsigned long long)rb+1);
  unsigned long long xs=(unsigned long long)xb*65536ull;
  __ESBMC_assert(rr <= xs, "u0.16: raw_r^2 <= raw_x * 2^16");
  __ESBMC_assert(nn > xs, "u0.16: (raw_r+1)^2 > raw_x * 2^16");
  return 0;}
