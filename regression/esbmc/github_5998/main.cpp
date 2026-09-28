typedef int data_t;
#define SIZE 12

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
  __ESBMC_assert(vec[6] <= max, "assertion after loop 6");
  __ESBMC_assert(vec[8] <= max, "assertion after loop 8");
  __ESBMC_assert(vec[10] <= max, "assertion after loop 10");
}
