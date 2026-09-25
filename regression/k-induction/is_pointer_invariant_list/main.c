// Linked-list traversal where the inductive step can only prove the
// assertion if `p`'s pre-havoc points-to set survives the IS havoc.
// Two loops: a build-loop that may grow the list, and a check-loop
// that walks it.  KNOWNBUG since #7964: the inductive step used to keep
// `p` inside its loop-entry objects, which proved this but also proved
// real bugs once `p` leaves them.  A sound proof needs the loop-head
// fixpoint of `p`'s points-to set rather than the loop-entry one.
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
  // walk's IS havocs of `p` lose the chain's identity and admit a model
  // where p->h != 1.
  List p = a;
  while (p) {
    if (p->h != 1)
      ERROR: { return 1; }
    p = p->n;
  }

  return 0;
}
