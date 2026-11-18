/* #86: the leak-free counterpart of github_86_fail under --schedule. */
#include <pthread.h>
#include <stdlib.h>

void *worker(void *arg)
{
  int *q = arg;
  *q = 1;
  return 0;
}

int main()
{
  pthread_t id;
  int *p = malloc(sizeof(int));
  pthread_create(&id, 0, worker, p);
  pthread_join(id, 0);
  free(p);
  return 0;
}
