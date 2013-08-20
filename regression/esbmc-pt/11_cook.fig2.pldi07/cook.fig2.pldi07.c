#include <pthread.h> 
int lock=0;
int x;


#define acquire_thread_id1(tid, l) \
{ __ESBMC_atomic_begin(); \
    __ESBMC_assume(l==0); \
    l = tid; \
    __ESBMC_atomic_end(); \
}


void thr1() {
  acquire_thread_id(1, lock); // lock=0 /\ lock'=1 
  while (x>0) {
    x = x-1;
  }
  release(lock);
}

void thr2() {
  int NONDET;
  while (NONDET) {
    acquire_thread_id(2, lock); // lock=0 /\ lock'=2 
    x = NONDET;
    release(lock);
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
