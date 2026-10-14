// Companion to github_5400: a node inserted into a single list (chosen by a
// non-deterministic index) keeps every allocation reachable from the global
// `slot[]`, so there is no leak. The global-reachability fixpoint must merge
// the inserted node's incoming path conditions across all candidate list
// heads; expanding it under only one path made its guard false on the other
// executions and produced a spurious "forgotten memory" leak. This pins the
// VERIFICATION SUCCESSFUL verdict (no false positive).
#include <stdlib.h>
extern int __VERIFIER_nondet_int();

struct s
{
  int datum;
  struct s *next;
};

struct s *mk(int x)
{
  struct s *p = malloc(sizeof(struct s));
  p->datum = x;
  p->next = 0;
  return p;
}

void list_add(struct s *node, struct s *list)
{
  struct s *t = list->next;
  list->next = node;
  node->next = t;
}

struct s *slot[3];

int main()
{
  int j = __VERIFIER_nondet_int();
  if (!(0 <= j && j < 3))
    return 0;

  for (int i = 0; i < 3; i++)
  {
    slot[i] = mk(1);
    list_add(mk(2), slot[i]);
  }

  list_add(mk(3), slot[j]); // fresh node into list j; everything stays reachable
  return 0;
}
