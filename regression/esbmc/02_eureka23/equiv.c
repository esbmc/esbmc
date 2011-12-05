
int f1(unsigned char a, unsigned char b) 
{
  int result=0, i;

  for(i=0; i<8; i++)
   if((b>>i)&1)
     result+=(a<<i);

//  assert(result==a*b);
  return result;
}


int f2(unsigned char a, unsigned char b) 
{
  int result=0, i;

  for(i=0; i<8; i++)
   if((b>>i)&1)
     result+=(a<<i);

  return result;
//  assert(result==a*b);
}

unsigned char nondet_uchar();

int main()
{
  unsigned char a, b;
  a=nondet_uchar();
  b=nondet_uchar();
  assert(f1(a,b)==f2(a,b));

  return 0;
}
