// Regression for issue #5400: under --no-reachable-memory-leak (SV-COMP
// valid-memtrack), a node inserted into two lists via a reused pointer orphans
// a previously-linked node. ESBMC used to report VERIFICATION SUCCESSFUL
// (unsound, missed leak) because the global-reachability fixpoint dropped a
// target's guard, treating the orphaned node as unconditionally reachable.
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
  for (int i = 0; i < 3; i++)
  {
    slot[i] = mk(1);
    list_add(mk(2), slot[i]);
  }

  int j = __VERIFIER_nondet_int(), k = __VERIFIER_nondet_int();
  if (!(0 <= j && j < 3))
    return 0;
  if (!(0 <= k && k < 3))
    return 0;

  struct s *p = mk(3);
  list_add(p, slot[j]);
  list_add(p, slot[k]); // reuse p across two lists -> orphans a node when j != k
  return 0;
}
