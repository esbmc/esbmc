// Positive companion for --cswitch-skip-readonly-globals.
//
// `config` is initialised once in main before any thread is created and is
// only ever READ by the worker threads — it is read-only across all threads,
// so its reads cannot participate in a data race. Each worker writes only its
// own disjoint global. With --cswitch-skip-readonly-globals the reads of
// `config` no longer force a context switch, yet the program stays safe and
// --data-races-check must NOT fire.
#include <pthread.h>
#include <assert.h>

int config = 7; // written once before spawn; read-only afterwards
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
  pthread_t a, b;
  pthread_create(&a, 0, worker_a, 0);
  pthread_create(&b, 0, worker_b, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(out_a == 8);
  assert(out_b == 9);
  return 0;
}
