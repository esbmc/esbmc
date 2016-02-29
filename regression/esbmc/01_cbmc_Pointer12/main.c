void f(const int *p)
{
  int *q;
  
  // this is ok
  q=(int *)p;
  
  // this, too!
  *q=1;
}

int main()
{
  int x;
  
  f(&x);
}
