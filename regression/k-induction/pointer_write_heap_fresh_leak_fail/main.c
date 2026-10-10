// Under a leak check the step has to model the nodes earlier iterations
// allocated, so the loop's nodes are not left unhavoced: that would let the
// inductive step prove that the two frees release every node. The step is
// disabled and the base case finds the leak.
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

struct node
{
  struct node *next;
};

int main()
{
  struct node *head = 0;
  while (__VERIFIER_nondet_int())
  {
    struct node *n = malloc(sizeof *n);
    __ESBMC_assume(n);
    n->next = head;
    head = n;
  }
  if (head)
  {
    struct node *s = head->next;
    free(head);
    free(s);
  }
}
