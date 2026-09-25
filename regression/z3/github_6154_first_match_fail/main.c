// GitHub #6154: pins "earliest matching return wins" for an early return inside
// a loop.  With vec == {5,5,6,7} the only reachable results of find are
// find(5)==0 (the FIRST match, not the second), find(6)==2, find(7)==3, and -1
// otherwise -- so no value yields 1 and the existential must fail.  A summarizer
// that let a later return override an earlier one would give find(5)==1 and
// wrongly report this SUCCESSFUL.
#define SIZE 4
typedef int data_t;

int find(data_t val, data_t vec[SIZE])
{
  for (int idx = 0; idx < SIZE; ++idx)
    if (vec[idx] == val)
      return idx;
  return -1;
}

int main()
{
  data_t vec[SIZE];
  vec[0] = 5;
  vec[1] = 5;
  vec[2] = 6;
  vec[3] = 7;
  data_t v;
  __ESBMC_assert(
    __ESBMC_exists(&v, find(v, vec) == 1), "no value first-matches at index 1");
  return 0;
}
