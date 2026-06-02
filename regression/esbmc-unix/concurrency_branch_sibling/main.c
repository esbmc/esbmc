// Execution-path coverage for record_branch_sibling +
// preserve_last_paths. The worker has a conditional branch on a
// shared variable; when the scheduler switches between the GOTO and
// the join, the sibling merge_statet pushed onto merge_state_map is
// captured by record_branch_sibling and survives the context switch
// via preserve_last_paths. After switching back, restore_last_paths
// re-emits the sibling and merge_gotos joins both paths.
//
// The assert is true on every interleaving; verification must
// succeed. This is breadth coverage — it confirms the path executes
// without crashing or aborting. A regression of record_branch_sibling
// (e.g. removing the hook) would manifest here as a verification
// failure or a missing sibling assertion.

#include <pthread.h>
#include <assert.h>

int flag = 0;
int val = 0;

void *worker(void *arg)
{
  if (flag)
    val = 1;
  else
    val = 2;
  return 0;
}

void *setter(void *arg)
{
  flag = 1;
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, worker, 0);
  pthread_create(&b, 0, setter, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(val == 1 || val == 2);
  return 0;
}
