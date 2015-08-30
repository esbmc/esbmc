void *malloc(unsigned s);

int main()
{
  int *p, *q;
  
  q=p=malloc(sizeof(int));
  __ESBMC_assume(p);
  
  *p=2;

  p=malloc(sizeof(int));
  __ESBMC_assume(p);
  
  *p=3;
  
  assert(*q==2);
  assert(*p==3);
}
