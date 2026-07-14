// Regression for constant propagation of struct values that contain a pointer
// field.
//
// `f` recurses only while `p->n >= 1`. The concrete structure built in main
// terminates the recursion at its real depth (2): root has n == 1 and points
// at leaf, leaf has n == 0 so the guard is false and recursion stops.
//
// Before the fix, ESBMC refused to constant-propagate a struct value whose
// WITH-chain updated a pointer-typed field (here `next`), because the pointer
// update was not classified as a plain constant. As a result `p->n` never
// folded, the recursion guard stayed symbolic, and symex explored the
// recursive branch on every path up to the --unwind bound -- tripping the
// recursion unwinding assertion at depth 3. With the fix the guard folds and
// the recursion stops at its real depth (2 < 3), so no unwinding assertion is
// reached.
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
  struct node leaf = {0, 0};
  struct node root = {1, &leaf};
  f(&root);
  return 0;
}
