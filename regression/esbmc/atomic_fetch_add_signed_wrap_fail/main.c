#include <limits.h>
#include <stdatomic.h>

atomic_int a = INT_MAX;

int main()
{
  int old = atomic_fetch_add(&a, 1);
  return old + 1;
}
