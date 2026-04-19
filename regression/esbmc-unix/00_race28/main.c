#include <pthread.h>
#include <assert.h>

int data1, data2;
void *t_fun(void *arg) {
  data1 = 5;
  return ((void *)0);
}
int main () {
  pthread_t id;
  pthread_create(&id,((void *)0),t_fun,((void *)0));
  data2 = 8;
  return 0;
}