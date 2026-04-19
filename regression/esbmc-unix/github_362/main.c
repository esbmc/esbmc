#include <pthread.h>
typedef struct {
  int b;
  pthread_mutex_t c;
} d;
typedef struct {
  d *e, f;
} g;
void* k(void* l) {
  g *a = l;
  a->e->b;
  return NULL;
}
int main() {
  d f;
  g a[2];
  for (int i = 0; i < 2; ++i)
    a[i].f = f;
  pthread_t j;
  pthread_create(&j, 0, k, a);
  return 0;
}

