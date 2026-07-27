int main(void)
{
  float x = 0.5f;
  float y = x * x;
  __ESBMC_assert(y > 0.0f, "0.5 * 0.5 = 0.25 is above the subnormal threshold");
  return 0;
}
