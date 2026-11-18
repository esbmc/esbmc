#include <cassert>
#include <pthread.h>
const int t[2] = {1, 2};
int g;
void *w(void *)
{
  for (int i = 0; i < 2; ++i)
    assert(t[i] == i + 1);
  return 0;
}
int main()
{
  const int *p = t;
  g = *p;
  pthread_t a, b;
  pthread_create(&a, 0, w, 0);
  pthread_create(&b, 0, w, 0);
  return 0;
}
