// __ESBMC_old(r[j]) with no __ESBMC_is_fresh extent stated: the snapshot
// covers the rest of the object r addresses. The body adds 1.
#define N 4

void nofresh(int *r)
{
  unsigned j;
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < N) || (r[j] == __ESBMC_old(r[j]) + 1)));
  __ESBMC_assigns(r);

  for (unsigned i = 0; i < N; i++)
    r[i] = r[i] + 1;
}

int main(void)
{
  int arr[N] = {1, 2, 3, 4};
  nofresh(arr);
  return 0;
}
