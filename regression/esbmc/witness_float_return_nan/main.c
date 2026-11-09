// No function_return waypoint for a NaN: \result == NAN holds for no value (#9979).
extern float __VERIFIER_nondet_float(void);
int main(void)
{
  float x = __VERIFIER_nondet_float();
  __ESBMC_assert(x == x, "target");
}
