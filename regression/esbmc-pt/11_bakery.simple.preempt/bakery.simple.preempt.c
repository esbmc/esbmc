#include <pthread.h>
int t1=0, t2=0; // N integer tickets
int x; // variable to test mutual exclusion

thr1() {
  int __temp_1__;
  while (1) {
    __temp_1__ = t2;
    t1 = __temp_1__ + 1;
    __temp_1__ = t2;
    while( t1 >= __temp_1__ && ( __temp_1__ > 0 ) ) {
      __temp_1__ = t2;
    }; // condition to exit the loop is (t1<t2 \/ t2=0)
    x = 0;
    assert(x <= 0);
    t1 = 0;
  }
}

thr2() {
  int __temp_2__;
  while (1) {
    __temp_2__ = t1;
    t2 = __temp_2__ + 1;
    __temp_2__ = t1;
    while( t2 >= __temp_2__ && ( __temp_2__ > 0 ) ) {
      __temp_2__ = t1;
    }; // condition to exit the loop is (t2<t1 \/ t1=0)
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
