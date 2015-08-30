int *global;

void *malloc(unsigned);

struct X
{
  int i;
  struct X *n;
};

int main()
{
  struct X *p;
  struct X x;
  int *q;
  
  p=malloc(sizeof(struct X));
  __ESBMC_assume(p);
  q=&(p->i);
  
  *q=1;
  
  assert(p->i==1);
}
