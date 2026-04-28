/* atomicity10: threads operate only on local variables.
   No global reads appear on RHS of any assignment, so the
   atomicity checker should find no violations: VERIFICATION SUCCESSFUL. */
#include <pthread.h>

void *t1(void *arg)
{
  int a = 1;
  int b = a + 2;
  (void)b;
  return NULL;
}

void *t2(void *arg)
{
  int c = 10;
  int d = c * 3;
  (void)d;
  return NULL;
}

int main(void)
{
  pthread_t id1, id2;
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  return 0;
}
