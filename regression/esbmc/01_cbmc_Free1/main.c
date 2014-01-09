void *malloc(unsigned);
void free(void *);

int main()
{
  int *p=malloc(sizeof(int));
  __ESBMC_assume(p);
  int *q=p;
  int i;
  
  if(i==4711) free(q);

  // should fail if i==4711
  *p=1;
}
