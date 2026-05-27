// Linked-list traversal where the inductive step can only prove the
// assertion if `p`'s pre-havoc points-to set survives the IS havoc.
// Two loops: a build-loop that may grow the list, and a check-loop
// that walks it.  Before the symex-side pointer-invariant rewrite,
// IS k=3 returned SAT (the deref-time encoding fell back to
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
  if (!a) return 0;
  a->h = 1;
  a->n = 0;

  // Build: optionally extend with more nodes whose `h` is also 1.
  List end = a;
  while (__VERIFIER_nondet_int()) {
    List t = (List)malloc(sizeof(struct node));
    if (!t) return 0;
    t->h = 1;
    t->n = 0;
    end->n = t;
    end = t;
  }

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
