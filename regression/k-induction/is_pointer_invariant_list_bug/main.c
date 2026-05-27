// Bug-finding companion to is_pointer_invariant_list. The build loop
// optionally appends a node whose `h` is 2 (not 1); the check loop
// asserts every node's h == 1.  BC at sufficient depth must find the
// counterexample. The symex-side pointer-invariant rewrite must NOT
// over-constrain the encoding into a spurious UNSAT here — every
// reachable model with h=2 must remain expressible.
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

  // Build: optionally extend with a node whose h is *2*.
  List end = a;
  while (__VERIFIER_nondet_int()) {
    List t = (List)malloc(sizeof(struct node));
    if (!t) return 0;
    t->h = 2;
    t->n = 0;
    end->n = t;
    end = t;
  }

  // Check: every node's h must be 1. The bug is reachable whenever
  // the build loop ran at least one iteration.
  List p = a;
  while (p) {
    if (p->h != 1)
      ERROR: { return 1; }
    p = p->n;
  }

  return 0;
}
