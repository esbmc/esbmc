// GitHub #6154: the reporter's program.  Merging an if/else inside an unrolled
// loop used to embed the running value in both mux arms, making the summary
// 2^n for a trip count of n; at SIZE=10 that tripped the size cap and the
// existential — which cannot be skolemized — was rejected outright.
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
  data_t a_vec[SIZE];
  idx_t a_idx_vfy;
  // Every element occurs at least once: the element at the witness index.
  __ESBMC_assert(
    __ESBMC_exists(
      &a_idx_vfy,
      (0 <= a_idx_vfy && a_idx_vfy < SIZE) &&
        count(a_vec[a_idx_vfy], a_vec) != 0),
    "some element occurs in the array");
  return 0;
}
