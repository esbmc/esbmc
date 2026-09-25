// GitHub #6154: a body that fits the default budget is rejected under a lowered
// --max-quantifier-summary-nodes, pinning both the option and the reason the
// rejection now reports.  Without a reason the same message covers a pointer
// write, a data-dependent trip count and an over-large summary alike.
#define SIZE 20
typedef char data_t;

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
  int i;
  __ESBMC_assert(
    __ESBMC_exists(&i, (0 <= i && i < SIZE) && count(vec[i], vec) != 0),
    "some element occurs in the array");
  return 0;
}
