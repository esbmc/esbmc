// Test for GitHub issue #3191 - C11 atomic types support
// This test should fail - array bounds violation with atomics
#include <stdatomic.h>
#include <assert.h>

int main()
{
  _Atomic int flag = 0;
  int data = 0;

  // Thread 1
  data = 42;
  atomic_store_explicit(&flag, 1, memory_order_release);

  // Thread 2
  if (atomic_load_explicit(&flag, memory_order_acquire) == 1)
  {
    assert(data == 42); // Guaranteed by acquire/release pair
  }

  return 0;
}
