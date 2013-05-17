#include <pthread.h>

int flag1 = 0, flag2 = 0; // N integer flags 
int x; // variable to test mutual exclusion

void thr1() {
  while (1) {
    flag1 = 1;
    while (flag2 >= 3); 
    flag1 = 3;
    if (flag2 == 1) {
      flag1 = 2;
      while (flag2 != 4);
    }
    flag1 = 4;
    while (flag2 >= 2);
    // begin critical section
    x = 0;
    assert(x<=0);
    // end critical section
    while (2 <= flag2 && flag2 <= 3);
    flag1 = 0;
  }
}

void thr2() {
  while (1) {
    flag2 = 1;
    while (flag1 >= 3);
    flag2 = 3;
    if (flag1 == 1) {
      flag2 = 2;
      while (flag1 != 4);
    }
    flag2 = 4;
    while (flag1 >= 2);
    // begin critical section
    x = 1;
    assert(x>=1);
    // end critical section
    while (2 <= flag1 && flag1 <= 3);
    flag2 = 0;
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}

