// #10154: a loop the state before it does enter is still proved by the
// inductive step.
int nondet_int();

int main()
{
  int i = nondet_int();
  int n = 4;
  for (; i < n; ++i)
    __ESBMC_assert(i < 5, "");
  return 0;
}
