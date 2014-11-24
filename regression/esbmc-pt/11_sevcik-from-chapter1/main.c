/* While this program cannot output value 1 in any interleaving
 optimising compilers often reuse the constant 1 assigned to 
 data in Thread 1 and replace the print data with print 1. */
#include <pthread.h> 
int requestReady = 0;
int responseReady = 0;
int data = 0;

void thr1() {
  data = 1;
  requestReady = 1;
  if (responseReady == 1) {
    // print data;
    assert (data != 1); // it is incorrect to print 1, because data might be different than 1
  }
}

void thr2() {
  if (requestReady == 1) {
    data = 2;
    responseReady = 1;
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
