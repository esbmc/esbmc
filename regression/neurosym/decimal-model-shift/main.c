/* The shift amount's bit pattern has its top bit set, so a decimal model
   renders it as a negative numeral. local_eval_bv() must mask every model
   value to the symbol's width before using it, or the guard on the shift
   arms (rhs >= width) never fires for it and mp_arith's operator<< reaches
   power(2, negative). */
struct wrap
{
  unsigned int f;
};

int main()
{
  unsigned int x = nondet_uint();
  unsigned int y = nondet_uint();
  struct wrap w;

  __ESBMC_assume(y > 0x80000000u);
  w.f = x << y;
  __ESBMC_assert(w.f != 0u, "shift by at least the width is non-zero");
  return 0;
}
