void foo(char **q)
{
  q[5];
}

int main(int argc, char **argv)
{
  int a;
  __ESBMC_assume(a >= 5 && a <= 20);
  char *b[++a];
  foo(b);
  return 0;
}