/* Out-of-bounds accesses: (*e).b[3] lies past the three ints the calloc paid
 * for, and a.b[0] reads storage the global `a` does not have, since a flexible
 * array member adds none (C17 6.7.2.1p18, #5393). */
#include<stdlib.h>
#include<assert.h>

struct A {
  char alloc;
  int b[];
} a;

int main() {
  struct A *e = calloc(1, sizeof(struct A) + 3*sizeof(int));
  (*e).b[3] = a.b[0];
  assert((*e).b[3] == 0);
  free(e);
}
