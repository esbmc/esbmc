#include <pthread.h> 
int g;

void thr() {
  int junk = 0;
  g = 0;
  junk++;
  junk++;
  junk++;
  g++;
  junk++;
  junk++;
  assert(g <=2);
  junk++;
  junk++;
  junk++;
  junk++;
  junk++;
}

int main() {
  pthread_t t1;
  pthread_create(&t1, NULL, thr, NULL);
  return 0;
}
