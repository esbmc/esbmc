// GitHub #6154: companion to github_6154_first_match_fail.  find(5, vec) is 0
// (the first of the two matches), so the existential holds at v == 5.
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
    __ESBMC_exists(&v, find(v, vec) == 0), "5 first-matches at index 0");
  return 0;
}
