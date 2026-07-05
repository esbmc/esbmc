// Pre-spawn runtime write, for --cswitch-skip-readonly-globals.
//
// Unlike the static-initialiser case, `config` here has no file-scope
// initialiser: it is written by a plain assignment in main's BODY, *before*
// main creates any thread. That write is single-threaded and cannot race, so
// `config` is still read-only from every worker's point of view. The spawn
// analysis (compute_thread_spawners / instructions_after_spawn) is what proves
// the write lies before the first spawn and keeps `config` optimisable — a
// static-init-only skip would not cover it.
//
// The workers only READ config and each writes its own disjoint global, so the
// program is race-free and --data-races-check must NOT fire.
#include <pthread.h>
#include <assert.h>

int config;     // no static initialiser: written at runtime in main's body
int out_a;      // written only by worker_a
int out_b;      // written only by worker_b

void *worker_a(void *_)
{
  out_a = config + 1;
  return 0;
}

void *worker_b(void *_)
{
  out_b = config + 2;
  return 0;
}

int main(void)
{
  config = 7; // runtime write in main's body, strictly before the first spawn
  pthread_t a, b;
  pthread_create(&a, 0, worker_a, 0);
  pthread_create(&b, 0, worker_b, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(out_a == 8);
  assert(out_b == 9);
  return 0;
}
