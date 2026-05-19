#include <pthread.h>

static pthread_mutex_t a = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t b = PTHREAD_MUTEX_INITIALIZER;

static void *t1(void *_)
{
  (void)_;
  pthread_mutex_lock(&a);
  pthread_mutex_lock(&b);
  pthread_mutex_unlock(&b);
  pthread_mutex_unlock(&a);
  return 0;
}

static void *t2(void *_)
{
  (void)_;
  pthread_mutex_lock(&b);
  pthread_mutex_lock(&a);
  pthread_mutex_unlock(&a);
  pthread_mutex_unlock(&b);
  return 0;
}

int main(void)
{
  pthread_t p1, p2;
  pthread_create(&p1, 0, t1, 0);
  pthread_create(&p2, 0, t2, 0);
  pthread_join(p1, 0);
  pthread_join(p2, 0);
  return 0;
}
