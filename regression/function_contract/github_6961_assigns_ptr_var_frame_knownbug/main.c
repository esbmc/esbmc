// A pointer variable argument reaches the havoc as a pointer, and a pointer
// names no object to widen to, so only *q is havocked. b[1] keeps its pre-call
// value even though the callee writes it, and the assertion below holds when it
// should not. Passing b directly (github_6961_assigns_ptr_param_fail) has the
// decay to widen and gets this right.
#define N 4

void clr(int *p)
{
  __ESBMC_assigns(p);
  __ESBMC_ensures(p[0] == 0);

  for (int i = 0; i < N; i++)
    p[i] = 0;
}

int main(void)
{
  int b[N];
  int *q = b;
  b[1] = 7;
  clr(q);
  __ESBMC_assert(b[1] == 7, "b[1] survived the replaced call");
  return 0;
}
