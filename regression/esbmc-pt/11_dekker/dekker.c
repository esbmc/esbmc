// #include <assert.h>

int flag1 = 0, flag2 = 0; // N boolean flags 
int turn = 0; // integer variable to hold the ID of the thread whose turn is it
int x; // variable to test mutual exclusion
#include <pthread.h>
void thr1() {
  flag1 = 1;
  while (flag2 >= 1) {
    if (turn != 0) {
      flag1 = 0;
      while (turn != 0) {};
      flag1 = 1;
    }
  }
  // begin: critical section
  x = 0;
  assert(x<=0);
  // end: critical section
  turn = 1;
  flag1 = 0;
}

void thr2() {
  flag2 = 1;
  while (flag1 >= 1) {
    if (turn != 1) {
      flag2 = 0;
      while (turn != 1) {};
      flag2 = 1;
    }
  }
  // begin: critical section
  x = 1;
  assert(x>=1);
  // end: critical section
  turn = 1;
  flag2 = 0;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}

