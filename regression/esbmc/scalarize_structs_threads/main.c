/* --scalarize-structs (#7521) leaves a program that may create threads alone:
 * a whole-struct copy split into one write per field would let another thread
 * see it half done. */
#include <pthread.h>

struct pair
{
  int x;
  int y;
};

struct pair shared;

void *writer(void *arg)
{
  shared.x = 1;
  shared.y = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  struct pair seen = shared;
  __ESBMC_assert(seen.x == seen.y || seen.x == 1, "x is written first");
  return 0;
}
