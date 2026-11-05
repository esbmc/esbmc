#include <stdlib.h>

struct S
{
  int a, b, c, d;
};

/* On ILP32 `ov.cur` and `apc.nc` sit at the same offset, and the
 * dereference below reads `cur` through `apc`. */
union Tail
{
  struct
  {
    union
    {
      struct
      {
        void *f, *b;
        unsigned k;
        unsigned char ins;
      } q;
      void *ctx[4];
    } u0;
    void *thread;
    void *aux;
    void *l1, *l2;
    struct S *cur;
  } ov;
  struct
  {
    short t, s;
    int sp;
    void *th;
    void *l1, *l2;
    void (*k)(void);
    void (*r)(void);
    void (*n)(void);
    void *nc;
    signed char idx, mode;
    unsigned char ins;
  } apc;
};

struct Irp
{
  int x;
  union Tail tail;
};

struct S big[4];
struct Irp irp;

int main(void)
{
  irp.tail.apc.nc = big;
  irp.tail.ov.cur = malloc(sizeof(struct S));
  if (!irp.tail.ov.cur)
    return 0;
  struct Irp *p = &irp;
  struct S *sp = p->tail.ov.cur;
  /* Out of bounds of the one-element block; in bounds of the stale `big`. */
  sp[1].a = 1;
  free(irp.tail.ov.cur);
  return 0;
}
