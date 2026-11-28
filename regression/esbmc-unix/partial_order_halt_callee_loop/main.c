#include <pthread.h>
#include <assert.h>
int flag = 0;
void *t(void *a) { flag = 1; return 0; }
void wait_flag(void) { while (flag == 0) ; }
int main() {
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  wait_flag();
  assert(flag == 1);
  return 0;
}
