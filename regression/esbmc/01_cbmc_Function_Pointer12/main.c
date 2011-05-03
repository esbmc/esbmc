int global;

typedef struct {
  int i;
} xt;

xt x;

void f(xt i)
{
  global=1;
}

void g(xt i)
{
  global=0;
}

int main()
{
  void (*p)(xt);
  _Bool c;
  
  p=c?f:g;
  
  x.i=2;

  p(x);
  
  assert(global==c);
}
