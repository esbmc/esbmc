/* main calls no spawn primitive: pthread_create is reached only by calling
 * through a function pointer. symex resolves such a call through the value set
 * (get_function_list in symex_function.cpp), so it can only reach a function
 * whose address is taken -- which is what the guard's worklist is seeded with.
 * Without that seeding this program reads as single-threaded and its loop is
 * cut. */
#include <pthread.h>

static void *body(void *arg)
{
  return 0;
}

static void spawn(void)
{
  pthread_t t;
  pthread_create(&t, 0, body, 0);
}

int main(void)
{
  void (*fp)(void) = spawn;
  unsigned int i = 0;
  unsigned int s = 0;
  while (i < 4)
  {
    s = s + 1;
    i = i + 1;
  }
  fp();
  __ESBMC_assert(s == 4, "s == 4");
  return 0;
}
