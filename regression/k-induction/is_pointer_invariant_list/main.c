// Linked-list traversal where the inductive step can only prove the
// assertion if the IS pointer-invariant strengthening is active.
// A build loop extends a list with nodes whose h == 1; a check loop
// walks it asserting h == 1.  Without value-set restore + SAME-OBJECT
// assume at the IS havoc, the deref encoding falls back to invalid_object
// and the solver finds a spurious counterexample.
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

typedef struct node
{
  int h;
  struct node *n;
} *List;

int main()
{
  List a = (List)malloc(sizeof(struct node));
  if (!a)
    return 0;
  a->h = 1;
  a->n = 0;

  List end = a;
  while (__VERIFIER_nondet_int())
  {
    List t = (List)malloc(sizeof(struct node));
    if (!t)
      return 0;
    t->h = 1;
    t->n = 0;
    end->n = t;
    end = t;
  }

  List p = a;
  while (p)
  {
    if (p->h != 1)
      ERROR:
      {
        return 1;
      }
    p = p->n;
  }

  return 0;
}
