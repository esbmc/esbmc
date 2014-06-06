void *malloc(unsigned s);

int main()
{
  int *p, *q;
  
  q=p=malloc(sizeof(int));
  __ESBMC_assume(q);
  
  *p=2;

  p=malloc(sizeof(int));
  __ESBMC_assume(p);
  
  *p=3;
  
  //this should fail
  assert(*q==3);
}
