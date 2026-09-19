// Linked-list traversal where the inductive step can only prove the
// assertion if `p`'s pre-havoc points-to set survives the IS havoc.
// The list is built without a loop (a loop writing heap nodes disables the
// inductive step) and may be cyclic, so the check-loop is unbounded and only
// the inductive step can close it.  Before the symex-side pointer-invariant
// rewrite, IS k=3 returned SAT (the deref-time encoding fell back to
// `invalid_object` for the walking `p`, making the assert violable).
// With the rewrite, IS k=3 proves it.
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

typedef struct node {
  int h;
  struct node *n;
} *List;

int main() {
  List a = (List)malloc(sizeof(struct node));
  List b = (List)malloc(sizeof(struct node));
  if (!a || !b) return 0;
  a->h = 1;
  b->h = 1;
  a->n = __VERIFIER_nondet_int() ? b : 0;
  b->n = __VERIFIER_nondet_int() ? a : 0;

  // Check: every node along the chain rooted at `a` has h == 1. The
  // walk's IS havocs of `p` would otherwise lose the chain's identity
  // and admit a model where p->h != 1; the pre-havoc value-set
  // restore prevents that.
  List p = a;
  while (p) {
    if (p->h != 1)
      ERROR: { return 1; }
    p = p->n;
  }

  return 0;
}
