// This example is runned first through the preprocessor 
// to replace the N symbol with a constant.
#include <pthread.h> 
int g, g1;

void thr() {
  glb_init(g>0);
  glb_init(g1==0);
  while (g1 < g) {
    g1 = g1 + 1;
  }
}

void main() {
  pthread_t t1;
  pthread_create(&t1, NULL, thr, NULL);
  assert(g1 <= g+__N__-1); // g1 <= g+N-1, where N is the number of threads ``thr''
}

