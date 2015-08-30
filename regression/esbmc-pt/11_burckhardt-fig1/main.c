// #include <assert.h>
#include <pthread.h> 

#define false 0
#define true 1

int B = 0;
int X = 0;
int Y = 0;

void* thr1() {
  int r = B;
  if (r) {
    X = r; 
    if (r) { Y = 0; } else { Y = 1; } // Y = !r;
  } else {
    if (r) { Y = 0; } else { Y = 1; } // Y = !r; 
    X = r;
  }
  return 0;
}

void* thr2() { // optimized version of thr1
  int r = B;
  X = r;
  if (r) { Y = 0; } else { Y = 1; } // Y = !r;
  return 0;
}

void* thr3() {
  X = 1;
  assert(X || Y);
  return 0;
}


int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  // pthread_create(&t1, NULL, thr2, NULL);
  pthread_create(&t2, NULL, thr3, NULL); 
  return 0;
}

