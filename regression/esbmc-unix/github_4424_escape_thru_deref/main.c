/* local escapes via a store THROUGH a pointer: *pp = &i */
#include <pthread.h>
int *g;
void *t_fun(void *a) { (*g)++; return 0; }      /* RACE */
int main(void) { pthread_t id; int i = 0; int **pp = &g; *pp = &i;
  pthread_create(&id, 0, t_fun, 0);
  i++;                                          /* RACE */
  pthread_join(id, 0); return 0; }
