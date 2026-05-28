// Execution-path coverage for restore_last_paths under preserved
// paths that converge at distinct instructions across a context
// switch. The worker has two chained if/else blocks, each pair
// joining at a distinct location, so merge_state_map accumulates
// multiple entries and the scheduler has multiple cswitch points
// to interleave.
//
// Note: this test does NOT tightly pin the .begin()->.back() fix in
// restore_last_paths — that bug is real but only flips verification
// outcomes on programs whose properties depend on the corrupted
// guard/num_instructions. Existing larger tests (e.g. dekker,
// pthread20) provide the actual regression coverage. This test
// exercises the code path so a future regression that aborts or
// crashes is caught.

#include <pthread.h>
#include <assert.h>

int flag1 = 0;
int flag2 = 0;
int sink = 0;

void *worker(void *arg)
{
  int a, b;
  if (flag1)
    a = 1;
  else
    a = 2;
  // First join.
  if (flag2)
    b = 10;
  else
    b = 20;
  // Second join.
  sink = a + b;
  return 0;
}

void *setter(void *arg)
{
  flag1 = 1;
  flag2 = 1;
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, worker, 0);
  pthread_create(&b, 0, setter, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  // sink ∈ {1+10, 1+20, 2+10, 2+20} = {11, 21, 12, 22}.
  assert(sink == 11 || sink == 12 || sink == 21 || sink == 22);
  return 0;
}
