/*
 * Issue #6189, multiple side-effecting invariants. Two invariants each call a
 * function whose loop is bounded by the havoc'd variable x, and a third, pure
 * invariant (x <= SIZE) supplies that bound. The pure conjunct must be assumed
 * before both calls so f(x) and g(x) are symex'd with x <= SIZE; the two calls
 * themselves keep their source order.
 */
#define SIZE 5

int f(int n)
{
  int r = 0;
#pragma unroll SIZE + 1
  for (int i = 0; i < n; ++i)
    ++r;
  return r;
}

int g(int n)
{
  int r = 0;
#pragma unroll SIZE + 1
  for (int i = 0; i < n; ++i)
    r += 2;
  return r;
}

int main()
{
  int x = 0;
  __ESBMC_loop_invariant(x <= SIZE);
  __ESBMC_loop_invariant(f(x) == x);
  __ESBMC_loop_invariant(g(x) == 2 * x);
  for (x = 0; x < SIZE; ++x)
  {
  }
  __ESBMC_assert(x == SIZE, "end");
}
