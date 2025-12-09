#include <pthread.h> 
int x, y;
int b1;

void thr1(){
  if (y){
    b1 = 0;
  }
}

void thr2(){
  x = 1;
  assert(x >= 1);
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
