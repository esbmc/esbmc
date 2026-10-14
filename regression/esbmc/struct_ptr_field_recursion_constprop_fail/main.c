// Failing companion to struct_ptr_field_recursion_constprop.
//
// The same recursion, but the concrete structure is a chain of four nodes that
// all have n >= 1, so the recursion legitimately runs deeper than the --unwind
// bound of 2. With the struct-pointer-field constant-propagation fix, ESBMC
// folds each `p->n` guard (every node's n is 1) and follows the chain exactly,
// reaching the unwind bound -- so the recursion unwinding assertion fails, which
// is the sound, expected outcome (VERIFICATION FAILED).
struct node
{
  int n;
  struct node *next;
};

void f(struct node *p)
{
  if (p == 0)
    return;
  if (p->n >= 1)
    f(p->next);
}

int main(void)
{
  struct node d = {1, 0};
  struct node c = {1, &d};
  struct node b = {1, &c};
  struct node a = {1, &b};
  f(&a);
  return 0;
}
