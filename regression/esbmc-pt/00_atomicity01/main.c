#include <pthread.h>
#include <assert.h>

int nondet_int();
int xl=nondet_int(), y=nondet_int(), z=nondet_int();

void* t1(void* arg) {
  xl=y+z;
  return NULL;
}

void* t2(void* arg) {
  z=2;
  return NULL;
}

int main(void) {
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}
