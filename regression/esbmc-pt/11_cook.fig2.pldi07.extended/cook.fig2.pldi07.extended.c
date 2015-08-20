#include <pthread.h> 
int lock=0;
int x;

void thr1() {
  int t;
  acquire(lock);
  while (x>0) {
    t = x-1;
    x = t;
  }
  release(lock);
}

void thr2() {
  int NONDET;
  while (NONDET) {
    acquire(lock);
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
