#include <limits.h>
#include <stdatomic.h>

atomic_int a = INT_MAX;
int b = INT_MAX;

int main()
{
  atomic_fetch_add(&a, 1);
  __atomic_fetch_add(&b, 1, __ATOMIC_SEQ_CST);
  __ESBMC_assert(a == INT_MIN && b == INT_MIN, "atomic add wraps");
  return 0;
}
