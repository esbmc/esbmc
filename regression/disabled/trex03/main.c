unsigned int nondet_uint();
_Bool nondet_bool();

int main()
{
  unsigned int x1=nondet_uint(), x2=nondet_uint(), x3=nondet_uint();
  unsigned int d1=1, d2=1, d3=1, kindice=0;
  _Bool c1=nondet_bool(), c2=nondet_bool();
  
  while(x1>0 && x2>0 && x3>0)
  {
    if (c1) x1=x1-d1;
    else if (c2) x2=x2-d2;
    else x3=x3-d3;
    c1=nondet_bool();
    c2=nondet_bool();
  }

  assert(x1==0 || x2==0 || x3==0);
  return 0;
}

