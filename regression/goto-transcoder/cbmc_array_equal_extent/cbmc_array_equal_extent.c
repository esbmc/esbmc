int main(void)
{
  int a[3] = {1, 2, 3};
  int b[2] = {1, 2};
  // CBMC answers false for arrays of different extent, but two spellings of one
  // type would compare unequal as ireps while CBMC calls them equal, so the
  // adapter declines a type mismatch rather than synthesising that false.
  __CPROVER_assert(!__CPROVER_array_equal(a, b), "different extent");
  return 0;
}
