typedef int data_t;
#define SIZE 150

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
  // Exercises a long running-max chain so a property against the *final*
  // result needs a direct leaf-to-final bound, not just per-step ones.
  __ESBMC_assert(vec[0] <= max, "assertion after loop 0");
  __ESBMC_assert(vec[149] <= max, "assertion after loop 149");
}
