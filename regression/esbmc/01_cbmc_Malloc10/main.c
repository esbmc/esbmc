void *malloc(unsigned s);

int main()
{
  int *p, *q;
  
  q=p=malloc(sizeof(int));
  
  *p=2;

  p=malloc(sizeof(int));
  
  *p=3;
  
  //this should fail
  assert(*q==3);
}
