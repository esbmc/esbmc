/* __CPROVER_havoc_object through a pointer value havocs only the objects the
 * pointer may land in, and claims nothing about them, a freed one included. */
void *malloc(__CPROVER_size_t);
void free(void *);
void __CPROVER_havoc_object(void *);
_Bool nondet_bool(void);

int g[2] = {1, 2};

int main(void)
{
  int *p = malloc(3 * sizeof(int));
  int *q = malloc(sizeof(int));
  *q = 1;
  __CPROVER_havoc_object(p + 1);
  p[0] = 4;
  __CPROVER_assert(p[0] == 4 && *q == 1, "only p's object is havocked");

  int x = 5, y = 6;
  int *xy = nondet_bool() ? &x : &y;
  __CPROVER_havoc_object(xy);
  __CPROVER_assert(xy == &x ? y == 6 : x == 5, "the other local is kept");

  int *h = g + 1;
  __CPROVER_havoc_object(h);
  free(q);
  __CPROVER_havoc_object(q);
  __CPROVER_havoc_object((int *)0);
  return 0;
}
