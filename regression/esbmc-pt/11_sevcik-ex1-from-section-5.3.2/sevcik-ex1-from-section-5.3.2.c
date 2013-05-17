/* The question is whether we can observe the result r2 = 1. This result is not
   possible in this program, but it becomes possible after rewriting r3=y to r3=r2. */
#include <pthread.h> 

int x = 0;
int y = 0;
int done1 = 0;

void thr1() {
  int r1;
  r1 = x;
  y = r1;
  done1 = 1;
}

void thr2() {
  int r2, r3;
  r2 = y;
  if (r2 == 1) {
    r3 = y;
    x = r3;
  } else { x = 1; }
  // assertion is provable, the result 1 is not possible
  if (done1 >= 1) { assert(r2 != 1); } 
}
  
int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
