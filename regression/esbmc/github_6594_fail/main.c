#include <pthread.h>
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);
extern void abort(void);

static void assume(int c)
{
  if (!c)
    abort();
}

struct s
{
  struct s *next;
};

struct s *slot[2];

static struct s *new_node(void)
{
  struct s *p = malloc(sizeof(struct s));
  p->next = 0;
  return p;
}

static void list_add(struct s *n, struct s *l)
{
  struct s *t = l->next;
  l->next = n;
  n->next = t;
}

static void *t_fun(void *a)
{
  (void)a;
  list_add(new_node(), slot[0]);
  return 0;
}

/* Same shape as github_6594, plus a node that is never linked into any list:
 * relaxing reachability far enough to accept the linked nodes must not accept
 * this one. */
int main(void)
{
  for (int q = 0; q < 2; q++)
    slot[q] = new_node();

  int j = __VERIFIER_nondet_int(), k = __VERIFIER_nondet_int();
  assume(0 <= j && j < 2);
  assume(0 <= k && k < 2);

  list_add(new_node(), slot[j]);
  list_add(new_node(), slot[k]);

  struct s *orphan = new_node();
  orphan = 0;
  (void)orphan;

  pthread_t t;
  pthread_create(&t, 0, t_fun, 0);
  return 0;
}
