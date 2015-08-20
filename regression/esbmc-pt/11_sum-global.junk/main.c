#include <pthread.h>
int g;
int g1 = 0;
int g2 = 0;

void thr1() {
  int junk = 0;
  junk++;
  junk++;
  junk++;
  junk++;
  junk++;
  while (g1 < g) {
    g1 = g1 + 1;
  }
  junk++;
  junk++;
  junk++;
  junk++;
  junk++;
}

void thr2() {
  int junk = 0;
  junk++;
  junk++;
  junk++;
  junk++;
  junk++;
  while (g2 < g) {
    g2 = g2 + 1;
  }
  junk++;
  junk++;
  junk++;
  junk++;
  junk++;
}

int main() {
  glb_init(g>0);
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thr1, NULL);
  pthread_create(&t2, NULL, thr2, NULL);
  assert(g1 <= g);
  assert(g2 <= g);
  assert(g1+g2 <= 2*g);
}
