int nondet_int();

typedef struct {
 int i;
} xt;

xt *x, **xx;

int main()
{
  xt y;

  x=&y;

  xx=&x;

  x->i=nondet_int();
  
  assert(x->i==2);

  return 0;
}
