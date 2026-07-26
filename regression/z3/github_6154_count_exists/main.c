// GitHub #6154: the same loop+if function inside __ESBMC_exists.  An existential
// cannot be skolemized, so this only holds when count() is genuinely summarized
// into a pure quantifier body.  vec == {1,1,2,3} makes count(1, vec) == 2.
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
  vec[0] = 1;
  vec[1] = 1;
  vec[2] = 2;
  vec[3] = 3;
  data_t v;
  __ESBMC_assert(
    __ESBMC_exists(&v, count(v, vec) == 2), "some value occurs twice");
  return 0;
}
