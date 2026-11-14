/* with_chain_linear, with the last element overwritten. */
#include <assert.h>

#define N 8000
int a[N];

int main()
{
  for (int i = 0; i < N; ++i)
    a[i] = i;
  a[N - 1] = 0;
  for (int i = 0; i < N; ++i)
    assert(a[i] == i);
  return 0;
}
