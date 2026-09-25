/* Deadlock-free: t1 releases m before it ever wants n, so no thread holds one
   mutex while waiting for the other and no wait-for cycle exists. The two
   critical sections are disjoint rather than nested, which is what separates
   this from github_6474_disjoint_fail.

   Counting blocked threads reports this as deadlocked if a blocked thread's
   contribution outlives the unlock that released it: the blocked count then
   reaches the running count and the assertion fires on a program that cannot
   deadlock. Cancelling a mutex's waiters at unlock time (#6474) is what keeps
   this SUCCESSFUL. */
#include <pthread.h>
pthread_mutex_t m, n;

void *t1(void *a)
{
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_lock(&n);
  pthread_mutex_unlock(&n);
  return 0;
}

void *t2(void *a)
{
  pthread_mutex_lock(&n);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_unlock(&n);
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_mutex_init(&m, 0);
  pthread_mutex_init(&n, 0);
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
