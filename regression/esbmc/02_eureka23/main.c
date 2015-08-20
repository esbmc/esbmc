
int main() 
{
  unsigned char a, b;
  int result=0, i;

  for(i=0; i<8; i++)
   if((b>>i)&1)
     result+=(a<<i);

  assert(result==a*b);
}
