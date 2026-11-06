// #9907: a pointer read through the larger union member must keep its object
#include <stdlib.h>
#include <string.h>
struct S { int a, b, c, d; };
union Tail {
  struct { union { struct { void *f, *b; unsigned k; unsigned char ins; } q; void *ctx[4]; } u0;
           void *thread; void *aux; void *l1, *l2; struct S *cur; } ov;
  struct { short t, s; int sp; void *th; void *l1, *l2; void (*k)(void); void (*r)(void);
           void (*n)(void); void *nc; signed char idx, mode; unsigned char ins; } apc;
};
struct Irp { int x; union Tail tail; };
void start(struct Irp *Irp)
{
  struct S *sp = Irp->tail.ov.cur, *next = Irp->tail.ov.cur - 1;
  memmove(next, sp, sizeof(int) * 2);
}
int main(void)
{
  struct Irp irp;
  irp.tail.ov.cur = malloc(4 * sizeof(struct S));
  if (!irp.tail.ov.cur) return 0;
  irp.tail.ov.cur += 3;
  start(&irp);
  free(irp.tail.ov.cur - 3);
  return 0;
}
