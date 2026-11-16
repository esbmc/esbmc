extern float __VERIFIER_nondet_float(void);
extern double __VERIFIER_nondet_double(void);
extern long double __VERIFIER_nondet_long_double(void);
int main(void)
{
  float x = __VERIFIER_nondet_float();
  double y = __VERIFIER_nondet_double();
  long double z = __VERIFIER_nondet_long_double();
  __ESBMC_assert(!(x == 255.95f && y == 1.0 / 0.0 && z == 3.25L), "target");
}
