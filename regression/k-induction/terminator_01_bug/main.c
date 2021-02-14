int nondet_int();

main()
{
  int x=nondet_int();
  int *p = &x;
 
  while(x<100) {
   (*p)++;
  }                       
  assert(0);    
  //  assert(array[0]>=menor);    
}

