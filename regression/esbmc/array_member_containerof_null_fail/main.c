#include <stddef.h>
#include <assert.h>
struct list { struct list *next; };
typedef struct { unsigned a; unsigned b; struct list l; } S;
#define container_of(ptr, T, m) ((T *)((char *)(ptr) - offsetof(T, m)))
S g;
int nondet_int(void);
int main(void)
{
  struct list *head = nondet_int() ? &g.l : (struct list *)0;
  S *p = container_of(head, S, l);
  struct list *l = &p->l;
  assert(l != 0);   /* fails when head == NULL */
  return 0;
}
