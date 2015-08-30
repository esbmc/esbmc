#include <stdlib.h>

int nondet_int();

struct S {
  int x;
  struct S *n;
};

int main(int argc, char *argv[]) {
  struct S *p, *q;

  p=malloc(sizeof(struct S));
  __ESBMC_assume(p);
  p->x = 5;
  p->n = NULL;
  
  argc=nondet_int();

  if (argc > 1) {
    q = malloc(sizeof(struct S));
    __ESBMC_assume(q);
    q->x = 5;
    q->n = p;
  } else {
    q = p;
  }
  
  assert(p->x + q->x == 10);
  
  free(q);
  free(p);
}
