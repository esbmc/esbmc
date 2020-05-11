void foo(char **q)
{
  q[2];
}

int main(int argc, char **argv)
{
  char *b[2+1];
  foo(b);
  return 0;
}