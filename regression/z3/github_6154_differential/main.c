// GitHub #6154: differential oracle.  `direct' is count() evaluated by ordinary
// symbolic execution of a real call; the quantified body is the summarized
// form of the same call.  Pinning them equal at the nondeterministic point c
// makes any divergence between the summarizer and real execution a test
// failure, rather than relying on a hand-computed expected value.
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
  data_t c;
  int direct = count(c, vec);
  data_t v;
  __ESBMC_assert(
    __ESBMC_forall(&v, v != c || count(v, vec) == direct),
    "summary agrees with real execution");
  return 0;
}
