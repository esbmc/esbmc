float nondet_float(void);

int main()
{
  float x = nondet_float();
  float max = 0.0f;
  if (x > max)
    max = x;
  __ESBMC_assert(x <= max, "post-merge bound should fail for NaN x");
  return 0;
}
