int nondet_int();
_Bool nondet_bool();

main()
{
  int x=nondet_int();
  int y=nondet_int();
  int z=nondet_int();

  while(x<100 && 100<z) 
  {
    _Bool tmp=nondet_bool();
    if (tmp)
   {
     x++;
   }
   else
   {
     x--;
     z--;
   }
  }                       
    
  assert(0);    
}


