int nondet_int();
_Bool nondet_bool();

main()
{
  int x=nondet_int();
  int y=nondet_int();

  while (x>0 && y>0)
  {
    _Bool c = nondet_bool();
    if(c) 
      x=x-1;
    else {
      x = nondet_int();
      y = y- 1;
    }
  }                           
  assert(0);    
}


