#include <pthread.h>
#include <assert.h>

int g = 0;
int *gp = &g;
int **gpp = &gp;

// Writes g through a nested dereference. get_expr_globals follows the pointer
// chain, so this is recorded against g and MPOR sees the conflict with main's
// direct write below (#6539).
void *writer(void *arg)
{
  *(*gpp) = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
