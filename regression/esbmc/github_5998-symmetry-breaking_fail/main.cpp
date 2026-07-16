typedef int data_t;
#define SIZE 8

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
  // vec[3] is singled out by an index-specific check, so it is NOT
  // interchangeable with the other cells; the flag must still catch this.
  if (vec[3] == 42)
    __ESBMC_assert(0, "position-specific bug must still be caught");
}
