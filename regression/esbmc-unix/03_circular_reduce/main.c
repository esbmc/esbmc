#include <assert.h>
#include <pthread.h>
_Bool receive = 0;
int i = 0;
void *t1() {
  for (int i = 0; i < 2; i++) {
    if (receive) {
      assert(i < 1);
      receive = 0;
    }
  }
  return NULL;
}
int main() {
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  receive = 1;
  return 0;
}
