#include <stdlib.h>

struct s
{
  int datum;
  struct s *next;
};

struct s *head_a, *head_b;

int nondet_int();

void *safe_malloc(size_t size)
{
  void *p = malloc(size);
  if (p == 0)
    abort();
  return p;
}

int main()
{
  int c = nondet_int();

  head_a = safe_malloc(sizeof(struct s));
  head_b = safe_malloc(sizeof(struct s));
  head_a->next = safe_malloc(sizeof(struct s));
  head_a->next->next = safe_malloc(sizeof(struct s));
  head_a->next->next->next = NULL;

  /* The third object hangs off whichever head the branch picks, so its
   * reachability -- and that of the fourth, which only it holds -- is the
   * disjunction of two routes, not just the one explored first. */
  if (c)
  {
    head_b->next = head_a->next;
    head_a->next = NULL;
  }
  else
    head_b->next = NULL;

  return 0;
}
