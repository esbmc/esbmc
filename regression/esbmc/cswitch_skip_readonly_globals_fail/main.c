// Soundness guard for --cswitch-skip-readonly-globals.
//
// `shared` is written by the worker thread, so it is NOT read-only and must
// keep triggering context switches even when the optimisation is enabled.
// The read in main races with the worker's write: the assertion can fail in
// the interleaving where the write lands first. ESBMC must report
// VERIFICATION FAILED with the flag on — proving the read-only filter never
// suppresses a real race.
#include <pthread.h>
#include <assert.h>

int shared = 0;

void *worker(void *_)
{
  shared = 1; // concurrent write -> `shared` may be written
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  int r = shared; // races with the worker's write
  assert(r == 0); // violated when the write happens first
  pthread_join(t, 0);
  return 0;
}
