typedef int data_t;
#define SIZE 6

int main()
{
  int idx;
  data_t vec[SIZE];
  data_t max = vec[0];
  //------------------------------------------------------------
  for (idx = 1; idx < SIZE; ++idx)
  {
    if (max < vec[idx])
      max = vec[idx];
  }
  //------------------------------------------------------------
  __ESBMC_assert(vec[0] <= max, "assertion after loop 0");
  __ESBMC_assert(vec[2] <= max, "assertion after loop 2");
  __ESBMC_assert(vec[4] <= max, "assertion after loop 4");
}
