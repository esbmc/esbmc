#include <pthread.h>

static int data1;
static int data2;
static pthread_rwlock_t rwlock;
void *t_fun(void *arg) {
  pthread_rwlock_rdlock(&rwlock);
  data1++;
  printf("%d",data2);
  pthread_rwlock_unlock(&rwlock);
  return ((void *)0);
}
int main(void) {
  pthread_t id;
  pthread_rwlock_init(&rwlock, NULL);
  pthread_create(&id, ((void *)0), t_fun, ((void *)0));
  pthread_rwlock_rdlock(&rwlock);
  printf("%d",data1);
  data2++;
  pthread_rwlock_unlock(&rwlock);
  pthread_join (id, ((void *)0));
  return 0;
}
