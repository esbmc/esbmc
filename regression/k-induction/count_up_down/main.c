unsigned int nondet_uint();

int main()
{
  unsigned int n = nondet_uint();
  __CPROVER_assume(n>0 && n<10000);
  int x=n, y=0;
  while(x>0)
  {
    x--;
    y++;
  }
  assert(y==n);
}
