// The referenced object's bits, not the reference: 1.0f is not 0.
static float g = 1.0f;
static float &ref()
{
  return g;
}

int main()
{
  __ESBMC_assert(__builtin_bit_cast(unsigned, ref()) == 0u, "lvalue ref");
  return 0;
}
