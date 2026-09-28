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

  /* Nothing takes over the third object on the branch that drops it, so both
   * it and the fourth are orphaned there. */
  if (c)
    head_a->next = NULL;
  else
    head_b->next = NULL;

  return 0;
}
