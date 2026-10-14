#define SIZE 3
typedef char data_t;
typedef int idx_t;

int main()
{
  idx_t idx;
  data_t vec[SIZE];
  idx_t idx_vfy;

  idx = 0;
  __ESBMC_loop_invariant(
    __ESBMC_forall(&idx_vfy, !(idx_vfy == 1) || (vec[idx_vfy] == 17)));

  while (idx < SIZE - 1)
  {
    vec[idx] = 3;
    ++idx;
  }
}
