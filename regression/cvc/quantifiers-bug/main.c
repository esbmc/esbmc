int zero_array[10];

int main()
{
  int sym;
  __ESBMC_assert(
    __ESBMC_forall(&sym, !(sym >= 0 && sym < 10) || zero_array[sym] == 0),
    "array is zero initialized");

  const unsigned N = 10;
  unsigned i = 0;
  char c[N];

  for (i = 0; i < N; ++i)
    c[i] = i;

  unsigned j;
  __ESBMC_assert(
    __ESBMC_forall(&i, i > 9 || c[i] == i), "array is initialized correctly");

  // BUG: esbmc simplifies i to N
  __ESBMC_assert(__ESBMC_exists(&i, i == 0), "can be anything");
  __ESBMC_assert(__ESBMC_exists(&i, i == 1), "can be anything 2");
  __ESBMC_assert(!__ESBMC_exists(&i, i == 0 && i == 1), "contradiction");
}
