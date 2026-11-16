// A float needs 9 significant digits and a double 17 to read back (#9979).
extern float __VERIFIER_nondet_float(void);
extern double __VERIFIER_nondet_double(void);
int main(void)
{
  float x = __VERIFIER_nondet_float();
  double y = __VERIFIER_nondet_double();
  __ESBMC_assert(!(x == 0x1.400016p+3f && y == 0x1.0000000000001p+0), "target");
}
