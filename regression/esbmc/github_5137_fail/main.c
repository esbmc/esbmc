// Companion to github_5137: a genuine write/write race on a large-offset array
// element. The packed race-flag index must still detect it (the fix must not
// mask real races at large offsets).
#include <pthread.h>

int arr[4096];

void *writer(void *p)
{
  arr[3000] = 1;
  return 0;
}

int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, writer, 0);
  pthread_create(&b, 0, writer, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
