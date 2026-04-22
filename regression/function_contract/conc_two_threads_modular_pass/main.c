/* conc_two_threads_modular_pass:
 * Modular concurrency verification with --replace-call-with-contract.
 *
 * Each thread calls write_val(int *p, int v) which writes *p = v.
 * The assigns(*p) clause enables PRECISE havoc: only *p is made nondet,
 * so the two threads do not interfere with each other's memory.
 *
 * Thread 1 owns &x (writes 10); Thread 2 owns &y (writes 20).
 * No data race (separate locations).  assert(x + y == 30) holds.
 *
 * Key insight: without __ESBMC_assigns, conservative havoc would clobber
 * ALL globals (including the other thread's variable), making the
 * assertion unprovable even though it is correct.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <pthread.h>
#include <assert.h>
#include <stddef.h>

int x = 0;
int y = 0;

void write_val(int *p, int v)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_assigns(*p);          /* precise: only the pointed-to int changes */
  __ESBMC_ensures(*p == v);
  *p = v;
}

void *t1(void *arg) { write_val(&x, 10); return NULL; }
void *t2(void *arg) { write_val(&y, 20); return NULL; }

int main()
{
  pthread_t a, b;
  pthread_create(&a, NULL, t1, NULL);
  pthread_create(&b, NULL, t2, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);
  assert(x + y == 30);
  return 0;
}
