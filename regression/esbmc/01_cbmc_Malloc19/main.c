void *malloc(unsigned);

void *my_malloc(unsigned size)
{
  void *p = malloc(size);
  __ESBMC_assume(p);
  return p;
}

struct S1
{
  int x;
};

struct S2
{
  char y;
};

int main(void)
{
  _Bool b;
  
  if(b)
  {
    struct S1 *p=my_malloc(sizeof(struct S1));
    p->x=1;
  }
  else
  {
    struct S2 *p=my_malloc(sizeof(struct S2));
    p->y=1;
  }
  
  return 0;
}
