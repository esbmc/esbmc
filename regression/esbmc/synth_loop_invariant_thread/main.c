/* Cutting a loop removes the interleaving points its body carried, so the
 * reader can no longer observe g == 1 or g == 2. The claim is downstream of
 * the havoc, so it is reported UNKNOWN rather than passing silently -- pinned
 * here because the alternative would be a false proof. Under BMC this is
 * VERIFICATION FAILED. */
#include <pthread.h>
#include <assert.h>

unsigned int g;

void *writer(void *arg)
{
  unsigned int i = 0;
  g = 0;
  while (i < 3)
  {
    g = g + 1;
    i = i + 1;
  }
  return 0;
}

void *reader(void *arg)
{
  assert(g == 0 || g == 3);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, writer, 0);
  pthread_create(&t2, 0, reader, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
