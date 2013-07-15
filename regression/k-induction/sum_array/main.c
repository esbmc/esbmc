
int nondet_int();

unsigned int nondet_uint();
unsigned int M = nondet_uint();

main()
{
  int A[M], B[M], C[M];
  unsigned int  i;
  
  for(i=0;i<M;i++)
    A[i] = nondet_int();
  
  for(i=0;i<M;i++)
    B[i] = nondet_int();

  for(i=0;i<M;i++)
     C[i]=A[i]+B[i];
  
  for(i=0;i<M;i++)
     assert(C[i]==A[i]+B[i]);
}

