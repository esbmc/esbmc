#include <pthread.h>
#include <stdio.h>

int myglobal;

void *t_fun(void *arg) {
  myglobal=myglobal+1; // RACE!
  return NULL;
}

int main(void) {
  pthread_t id;
  pthread_create(&id, NULL, t_fun, NULL);
  scanf("%d", &myglobal); // RACE!
  return 0;
}
