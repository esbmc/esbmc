#include <pthread.h>
#include <stdlib.h>

struct priv
{
  void *data;
};

struct dev
{
  struct priv *p;
};

struct priv shared;

void *worker(void *arg)
{
  shared.data = arg;
  return 0;
}

int main(void)
{
  struct dev *d = malloc(sizeof(struct dev));
  d->p = &shared;
  pthread_t id;
  pthread_create(&id, 0, worker, d);
  void *x = d->p->data;
  pthread_join(id, 0);
  return x != 0;
}
