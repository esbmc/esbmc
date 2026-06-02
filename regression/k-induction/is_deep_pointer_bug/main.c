// Regression for issue #5025: the k-induction inductive-step pointer
// invariant in #4838 (and the all-concrete gate in #5027 alone) was unsound
// because it read the symex-prefix value-set rather than a goto-program-level
// fixpoint. This test exercises the failure mode #4838 lacked a test for:
//
//   - a linked list of 5 nodes, last one (h=2) violates the traversal check;
//   - the violation depth is 5, deeper than --max-inductive-step 3 reaches in
//     the base case, so the verdict at this k must come from IS, not BC;
//   - with #4838's unsound rewrite, IS converged over the restricted candidate
//     set and reported VERIFICATION SUCCESSFUL on this bug;
//   - with the goto-program-level VSA fixpoint as the candidate source, the
//     reachable iteration that points p at the 5th node is in the over-
//     approximation, so the assume does not exclude the bug.
//
// The desc accepts FAILED or UNKNOWN as passing (precise verdict is solver-
// and strategy-dependent). The unsound case is SUCCESSFUL, which the regex
// must never match.
#include <stdlib.h>

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

  // Build 4 more nodes; the 4th (5th in the chain) has h == 2.
  List cur = a;
  for (int i = 1; i <= 4; i++)
  {
    List t = (List)malloc(sizeof(struct node));
    if (!t)
      return 0;
    t->h = (i == 4) ? 2 : 1;
    t->n = 0;
    cur->n = t;
    cur = t;
  }

  // Traverse: the assertion is violated at the 5th node.
  List p = a;
  while (p)
  {
    if (p->h != 1)
ERROR: { return 1; }
    p = p->n;
  }

  return 0;
}
