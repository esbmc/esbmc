#include <pthread.h>
int choosing1 = 0, choosing2 = 0; // N boolean flags
int number1 = 0, number2 = 0; // N natural numbers
int x; //variable to test mutual exclusion

void* thr1() {
  int tmp;
  choosing1 = 1;
  tmp = number2 + 1;
  number1 = tmp;
  choosing1 = 0;
  while (choosing2 >= 1) {};
  while (number1 >= number2 && number2 > 0) {
    // condition to exit the loop is (number1<number2 \/ number2=0)
  }
  // begin: critical section
  x = 0;
  assert(x <= 0);
  // end: critical section
  number1 = 0;
  pthread_exit(NULL);
}

void* thr2() {
  int tmp;
  choosing2 = 1;
  tmp = number1 + 1;
  number2 = tmp;
  choosing2 = 0;
  while (choosing1 >= 1) {};
  while (number2 >= number1 && number1 > 0) {
    // condition to exit the loop is (number2<number1 \/ number1=0)
  }
  // begin: critical section
  x = 1;
  assert(x >= 1);
  // end: critical section
  number2 = 0;
  pthread_exit(NULL);
}

int main() 
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, thr1, NULL);
  pthread_create(&id2, NULL, thr2, NULL);
  return 0;
}
