// GitHub #6154: a function with a bounded loop and an if inside a quantifier.
// count(v, vec) is summarized (loop unrolled, if muxed) into a pure expression,
// so the forall is a genuine quantifier over the fresh bound variable.
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
  data_t v;
  __ESBMC_assert(__ESBMC_forall(&v, count(v, vec) <= SIZE), "count bounded");
  return 0;
}
