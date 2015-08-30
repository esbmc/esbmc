#include <pthread.h> 
int x, y;

void thr1() {
  while (x>0 && y>0) {
    { __blockattribute__((atomic))
      x = x-1;
      y = x;
    } 
  }
}    

void thr2() {
  while (x>0 && y>0) {
    { __blockattribute__((atomic))
      x = y-2;
      y = x+1;
    } 
  }
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  return 0;
}
