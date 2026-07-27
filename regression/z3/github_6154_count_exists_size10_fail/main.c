// GitHub #6154: negative counterpart of github_6154_count_exists_size10.  A
// summary that collapsed to something vacuous would report SUCCESSFUL here too,
// so this pins that the summarized count() carries its real value: over a
// concrete array where 7 occurs four times, no index yields a count of five.
#define SIZE 10
typedef int idx_t;
typedef char data_t;

int count(data_t val, data_t vec[SIZE])
{
  idx_t idx;
  int res = 0;
  for (idx = 0; idx < SIZE; ++idx)
    if (vec[idx] == val)
      ++res;
  return res;
}

int main()
{
  data_t a_vec[SIZE] = {7, 7, 7, 1, 2, 3, 4, 5, 6, 7};
  idx_t a_idx_vfy;
  __ESBMC_assert(
    __ESBMC_exists(
      &a_idx_vfy,
      (0 <= a_idx_vfy && a_idx_vfy < SIZE) &&
        count(a_vec[a_idx_vfy], a_vec) == 5),
    "no element occurs five times");
  return 0;
}
