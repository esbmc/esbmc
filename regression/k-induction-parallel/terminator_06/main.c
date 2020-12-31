int nondet_int();
_Bool nondet_bool();

main()
{
  int x, y, z;

  while (x>0 && y>0 && z>0)
  {
    _Bool c1 = nondet_bool();
    _Bool c2 = nondet_bool();
    if(c1) {
      x=x-1;
    } else if (c2) {
      y = y- 1;
      z=nondet_int();
    } else {
      z=z-1;
      x=nondet_int();
    }
  }                           
  assert(!(x>0 && y>0 && z>0));    
}


