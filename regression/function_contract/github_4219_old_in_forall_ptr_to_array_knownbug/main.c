// `int (*r)[N]` does name a whole array, so the lift fires and takes the
// non-member path, but the snapshot it builds reads that array through a
// dereference and the dereference layer refuses an array rvalue:
//   ERROR: Can't construct rvalue reference to array type during dereference
// The struct-member path (github_4219_old_in_forall) avoids this by taking the
// struct instead and re-applying the member; a pointer to array has no such
// enclosing object to take.
#define N 4
#define BOUND 100

void bump(int (*r)[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(int[N])));
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] > -BOUND && (*r)[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] == __ESBMC_old((*r)[j]) + 1)));
  __ESBMC_assigns(*r);

  for (unsigned i = 0; i < N; i++)
    (*r)[i] = (*r)[i] + 1;
}

int main(void)
{
  return 0;
}
