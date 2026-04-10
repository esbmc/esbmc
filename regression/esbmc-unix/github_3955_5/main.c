/* Both-sides atomic: a = b splits into an atomic load of b followed by an
 * atomic store to a, with a context-switch window between them.
 * No data race must be reported.  Also verifies that reading an _Atomic
 * variable twice in one expression (b + b) produces only one atomic load
 * (deduplication: collect_atomic_reads uses a map keyed on identifier). */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int a = 0;
atomic_int b = 0;

void *t1_body(void *arg)
{
  b = 5;   /* atomic store */
  a = b;   /* atomic load of b, then atomic store to a — two separate ops */
  return NULL;
}

void *t2_body(void *arg)
{
  /* read b + b: must produce exactly one atomic load of b (deduplication) */
  int v = b + b; /* CK_AtomicToNonAtomic twice, but one tmp */
  (void)v;
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, t1_body, NULL);
  pthread_create(&t2, NULL, t2_body, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  int result = a; /* CK_AtomicToNonAtomic */
  assert(result == 0 || result == 5);
  return 0;
}
