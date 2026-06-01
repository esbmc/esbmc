// Soundness pin: same shape as is_pointer_invariant_list but the build
// loop appends a node with h == 2.  The IS pointer-invariant strengthening
// must NOT hide this genuine bug — VERIFICATION FAILED is required.
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
    t->h = 2;
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
