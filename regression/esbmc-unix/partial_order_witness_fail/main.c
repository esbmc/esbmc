/* #10990: the witness orders the threads' waypoints by the event clocks. */
#include <assert.h>
#include <pthread.h>
void __VERIFIER_atomic_begin(void);
void __VERIFIER_atomic_end(void);
int i = 3, j = 6;
void *t1(void *arg) {
  for (int k = 0; k < 3; k++) { __VERIFIER_atomic_begin(); i = j + 1; __VERIFIER_atomic_end(); }
  pthread_exit(0);
}
void *t2(void *arg) {
  for (int k = 0; k < 3; k++) { __VERIFIER_atomic_begin(); j = i + 1; __VERIFIER_atomic_end(); }
  pthread_exit(0);
}
int main(void) {
  pthread_t id1, id2;
  pthread_create(&id1, 0, t1, 0);
  pthread_create(&id2, 0, t2, 0);
  __VERIFIER_atomic_begin(); int condI = i >= (2*3 + 6); __VERIFIER_atomic_end();
  __VERIFIER_atomic_begin(); int condJ = j >= (2*3 + 6); __VERIFIER_atomic_end();
  assert(!condI && !condJ);
  return 0;
}
