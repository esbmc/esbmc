#include <pthread.h> 
int x=1;
int y=0;

void thr1() {
  while (x=1) {
    y = y+1;
  }
  while (y>0) {
    y = y-1;
  }
}

void thr2() {
  x = 0;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
