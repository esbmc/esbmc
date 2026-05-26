// Regression for #4634.
//
// Two-element linked list `A -> p`. A spawned thread inserts a new
// node between them by saving the old `A->next` in a stack local `t`,
// then overwriting `A->next`. Until the thread completes, the global
// head `A` no longer reaches the original `p` directly, but `t` does —
// at every schedule the program is memory-safe (the join pulls the
// thread to completion before main returns).
//
// Before #4634, `add_memory_leak_checks()` rooted its reachability
// constraint at globals only, so the intermediate schedule where the
// thread is suspended between `A->next = new_node` and the splice
// back-link was reported as a forgotten-memory violation whenever the
// `main_thread_ended` throttle was bypassed (e.g. `--data-races-check`).

#include <pthread.h>
#include <stdlib.h>

struct s
{
  int datum;
  struct s *next;
};

static struct s *A;
static pthread_mutex_t A_mutex = PTHREAD_MUTEX_INITIALIZER;

static void init(struct s *p, int x)
{
  p->datum = x;
  p->next = NULL;
}

static void *t_fun(void *arg)
{
  (void)arg;
  struct s *p = malloc(sizeof(struct s));
  if (!p)
    return NULL;
  struct s *t;
  init(p, 7);

  pthread_mutex_lock(&A_mutex);
  t = A->next;
  A->next = p;
  p->next = t;
  pthread_mutex_unlock(&A_mutex);
  return NULL;
}

int main(void)
{
  pthread_t t1;
  struct s *p = malloc(sizeof(struct s));
  if (!p)
    return 0;
  init(p, 9);

  A = malloc(sizeof(struct s));
  if (!A)
    return 0;
  init(A, 3);
  A->next = p;

  pthread_create(&t1, NULL, t_fun, NULL);
  pthread_join(t1, NULL);
  return 0;
}
