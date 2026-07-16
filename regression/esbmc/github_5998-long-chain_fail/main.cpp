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
  // vec[100] is singled out by an index-specific check on a long chain, so
  // the direct leaf-to-final bound must not mask this genuine bug.
  if (vec[100] == 42)
    __ESBMC_assert(0, "position-specific bug must still be caught");
}
