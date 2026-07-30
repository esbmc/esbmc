#define SIZE 10
typedef int idx_t;
typedef char data_t;
int main()
{
  data_t vec[SIZE];
  idx_t prv_idx = SIZE - 1;
  idx_t cur_idx = 0;
  idx_t nxt_idx;
  int loop_idx_vfy = 0;
  while (cur_idx != prv_idx)
  {
    if (vec[cur_idx] == 3)
      nxt_idx = (cur_idx + prv_idx) / 2;
    else
      nxt_idx = 0;
    prv_idx = cur_idx;
    cur_idx = nxt_idx;
    ++loop_idx_vfy;
  }
  __ESBMC_assert(cur_idx == prv_idx, "end: idxs equal");
}
