int g;

void bump(int *p)
  __CPROVER_assigns(*p)
{
  *p = 1;
  g = 2; /* not declared in the assigns clause */
}

int main()
{
  int x = 0;
  bump(&x);
  return 0;
}
