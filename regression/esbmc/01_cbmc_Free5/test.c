#include <stdlib.h>

int nondet_int();

struct S {
  int x;
  struct S *n;
};

int main(int argc, char *argv[]) {
  int *p, *q, i=nondet_int();

  p = &i;
  
  argc=nondet_int();

  if (argc > 1) {
    q = &i;
  } else {
    q = p;
  }
  
  assert(*(p) + *(q) == 2*i);
  
}
