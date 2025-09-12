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
    __ESBMC_forall(&j, j > 9 || c[j] == j), "array is initialized correctly");

  int value;
  __ESBMC_assert(__ESBMC_exists(&value, value == 0), "can be anything");
  __ESBMC_assert(__ESBMC_exists(&value, value == 1), "can be anything 2");
  __ESBMC_assert(
    !__ESBMC_exists(&value, value == 0 && value == 1), "contradiction");
}
