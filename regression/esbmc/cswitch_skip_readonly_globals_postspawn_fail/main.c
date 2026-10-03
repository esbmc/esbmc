// Post-spawn soundness guard for --cswitch-skip-readonly-globals.
//
// `shared` has no static initialiser and its only write in main's body runs
// *after* pthread_create -- so it is concurrent with the worker's read, not a
// single-threaded pre-spawn write. The spawn analysis must therefore NOT
// exclude it: instructions_after_spawn has to report the write as reachable
// from the spawn (it lexically follows pthread_create) and keep `shared`
// written, so its reads still trigger context switches.
//
// The worker's read races with main's write: in the interleaving where main
// writes first, the assertion fails. ESBMC must report VERIFICATION FAILED
// with the flag on. If the spawn analysis wrongly excluded the post-spawn
// write, `shared` would be treated as read-only and the race would be hidden.
#include <pthread.h>
#include <assert.h>

int shared;

void *worker(void *_)
{
  int r = shared; // races with main's post-spawn write
  assert(r == 0); // violated when main's write lands first
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  shared = 1; // write AFTER spawn -> concurrent, must stay race-visible
  pthread_join(t, 0);
  return 0;
}
