/* Only NaN violates the claim; one witness blocks every NaN payload. */
float nondet_float(void);
int main(void)
{
  float x = nondet_float();
  __ESBMC_assert(x == x, "x is not NaN");
  return 0;
}
