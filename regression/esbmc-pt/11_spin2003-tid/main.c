#include <pthread.h> 
int x=1, m=0;

#define acquire_thread_id(tid, l) \
  { __ESBMC_atomic_begin(); \
    __ESBMC_assume(l==0); \
    l = tid; \
    __ESBMC_atomic_end(); \
  } \

void thr1() {
  acquire_thread_id(1, m); // m=0 /\ m'=1
  x = 0;
  x = 1;
  assert(x>=1);
  release(m);
}

void thr2() {
  acquire_thread_id(2, m); // m=0 /\ m'=2
  x = 0;
  x = 1;
  assert(x>=1);
  release(m);
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}

