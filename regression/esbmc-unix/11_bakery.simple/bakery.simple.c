#include <pthread.h>
int t1=0, t2=0; // N natural-number tickets
int x; // variable to test mutual exclusion

thr1() {
  while (1) {
    t1 = t2 + 1;
    while( t1 >= t2 && ( t2 > 0 ) ) {}; // condition to exit the loop is (t1<t2 \/ t2=0)
    x=0;
    assert(x <= 0);
    t1 = 0;
  }
}

thr2() {
  while (1) {
    t2 = t1 + 1;
    while( t2 >= t1 && ( t1 > 0 ) ) {}; // condition to exit the loop is (t2<t1 \/ t1=0)
    x = 1;
    assert(x >= 1);
    t2 = 0;
  }
}

int main() 
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, thr1, NULL);
  pthread_create(&id2, NULL, thr2, NULL);
  return 0;
}
