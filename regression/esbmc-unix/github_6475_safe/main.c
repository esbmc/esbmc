#include <pthread.h>

static pthread_rwlock_t l;
static int data;

static void *reader(void *p)
{
  (void)p;
  pthread_rwlock_rdlock(&l);
  (void)data;
  pthread_rwlock_unlock(&l);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_rwlock_init(&l, 0);
  pthread_rwlock_wrlock(&l);
  data = 1;
  pthread_create(&t1, 0, reader, 0);
  pthread_create(&t2, 0, reader, 0);
  pthread_rwlock_unlock(&l);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
