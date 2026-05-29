// Companion to github_4425_funptr_race: the write to the function pointer and
// the indirect call through it are now guarded by the SAME mutex, so the
// accesses are mutually exclusive and there is no data race.  This confirms
// that instrumenting the function-pointer read at the call site does not raise
// a spurious race when the program is correctly synchronised.
//
// Expected: VERIFICATION SUCCESSFUL.
#include <pthread.h>

int f1(void)
{
  return 4;
}
int f2(void)
{
  return 5;
}

int (*fp)(void) = f1;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  pthread_mutex_lock(&mutex);
  fp = f2;
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, NULL, t_fun, NULL);
  pthread_mutex_lock(&mutex);
  fp();
  pthread_mutex_unlock(&mutex);
  pthread_join(id, NULL);
  return 0;
}
