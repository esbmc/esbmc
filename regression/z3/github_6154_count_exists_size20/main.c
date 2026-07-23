// GitHub #6154: twice the trip count of the reported cliff.  The pre-fix
// summary needed 2^20 nodes here, so this pins that the branch merge no longer
// grows exponentially rather than that the cap was merely nudged upwards.
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
    __ESBMC_exists(
      &i, (0 <= i && i < SIZE) && count(vec[i], vec) != 0),
    "some element occurs in the array");
  return 0;
}
