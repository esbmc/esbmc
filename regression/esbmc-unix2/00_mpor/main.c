#include <pthread.h>
#include <assert.h>

int join1 = 0;

void* t1(void* arg) {
  join1 = 1;
  return NULL;
}

int main(void) {
  pthread_t id1;
  pthread_create(&id1, NULL, t1, NULL);
  if(join1)
    assert(!join1);
  return 0;
}
