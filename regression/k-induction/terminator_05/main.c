int nondet_int();
_Bool nondet_bool();

main()
{
  int x, y, d;

  while (x>0 && y>0 && d>0)
  {
    _Bool c = nondet_bool();
    if(c) {
      x=x-1;
      d=nondet_int();
    } else {
      x = nondet_int();
      y = y- 1;
      d=d-1;
    }
  }                           
  assert(!(x>0 && y>0 && d>0));    
}


