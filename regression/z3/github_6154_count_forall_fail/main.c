// GitHub #6154: discriminating failing case.  With every element equal to 7,
// count(7, vec) == SIZE, so the claim count(v, vec) <= SIZE-1 is false for v==7.
#define SIZE 4
typedef int data_t;

int count(data_t val, data_t vec[SIZE])
{
  int res = 0;
  for (int idx = 0; idx < SIZE; ++idx)
    if (vec[idx] == val)
      ++res;
  return res;
}

int main()
{
  data_t vec[SIZE];
  for (int i = 0; i < SIZE; i++)
    vec[i] = 7;
  data_t v;
  __ESBMC_assert(
    __ESBMC_forall(&v, count(v, vec) <= SIZE - 1), "should fail at v==7");
  return 0;
}
