#include <assert.h>
#include <pthread.h>

_Bool receive = 0;

void *t1(void *arg)
{
  // The racy schedule is: i=0 reads 0 (skip), main writes 1, i=1 reads 1 and
  // the assertion fires -- before `receive = 0` ever runs. Adding that write
  // nonetheless loses the counterexample; writing any other variable, or
  // writing `receive = 1`, keeps it (#6558). See race_guard_other_write.
  for (int i = 0; i < 2; i++)
    if (receive)
    {
      assert(i < 1);
      receive = 0;
    }
  return NULL;
}

int main()
{
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  receive = 1;
  return 0;
}
