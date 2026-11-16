// #664: clang's diagnostics must follow --file-output instead of stderr.
int nondet_int();

int main()
{
  int x = nondet_int();
  if (x = 0)
    return 1;
  __ESBMC_assert(x == 0, "x was assigned 0");
  return 0;
}
