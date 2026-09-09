/* A false ensures over fixed-point types must be refuted: halving 0 does
 * not strictly decrease it. Keeps contract enforcement honest on fixed
 * types (a vacuous encoding would verify this). */
short _Fract halve(short _Fract x)
{
  __ESBMC_requires(x >= 0.0hr);
  __ESBMC_ensures(__ESBMC_return_value < x);
  return x >> 1;
}

int main(void)
{
  return halve(0.0hr) > 0.0hr;
}
