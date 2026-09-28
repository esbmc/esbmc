/* escaped local reached through two pointer levels */
#include <pthread.h>
pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER, m2 = PTHREAD_MUTEX_INITIALIZER;
void *t_fun(void *arg) {
  int **pp = (int **)arg;
  pthread_mutex_lock(&m1);
  (**pp)++;          /* RACE */
  pthread_mutex_unlock(&m1);
  return 0;
}
int main(void) {
  pthread_t id; int i = 0; int *p = &i;
  pthread_create(&id, 0, t_fun, (void *)&p);
  pthread_mutex_lock(&m2);
  i++;               /* RACE */
  pthread_mutex_unlock(&m2);
  pthread_join(id, 0);
  return 0;
}
