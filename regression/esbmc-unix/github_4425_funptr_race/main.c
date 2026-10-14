#include <pthread.h>

/* Regression for esbmc/esbmc#4425 (goblint-regression/04-mutex_50-funptr_rc).
 *
 * The global function pointer `fp` is written by t_fun under mutex1 and read by
 * main (via the indirect call `fp()`) under mutex2. Because the two accesses use
 * different locks they are unsynchronized -> a W/R data race on `fp`.
 *
 * The read of `fp` happens through the indirect call target `*fp`. rw_sett did
 * not record reads of an indirect-call target, so only the write was
 * instrumented (seen as a W/W access) and the race was missed (VERIFICATION
 * SUCCESSFUL on a racy program).
 */

int f1(void)
{
  return 4;
}
int f2(void)
{
  return 5;
}

int (*fp)(void) = f1;

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&mutex1);
  fp = f2; /* RACE: write under mutex1 */
  pthread_mutex_unlock(&mutex1);
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t_fun, 0);
  pthread_mutex_lock(&mutex2);
  fp(); /* RACE: read of fp under a different lock (mutex2) */
  pthread_mutex_unlock(&mutex2);
  pthread_join(id, 0);
  return 0;
}
