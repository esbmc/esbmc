#include <pthread.h>
#include <assert.h>

int g;
int *p = &g; // g's address escapes -> g is address-taken

void *writer(void *_)
{
  *p = 1; // unresolved pointer write -> any_indirect_write, target is g
  return 0;
}

void *reader(void *_)
{
  int a = g;
  int b = g;
  assert(a == b); // fails when the writer runs between the two loads
  return 0;
}

int main(void)
{
  pthread_t w, r;
  pthread_create(&w, 0, writer, 0);
  pthread_create(&r, 0, reader, 0);
  pthread_join(w, 0);
  pthread_join(r, 0);
  return 0;
}
