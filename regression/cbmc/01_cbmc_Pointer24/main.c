char a[100];

int main()
{
  char *p, *q;
  
  q=p;
  
  __ESBMC_assume(!__ESBMC_same_object(p, 0));
  
  p++;
  
  assert(!__ESBMC_same_object(p, 0));
}
