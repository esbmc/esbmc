//#include <stdio.h>

int gcd(int a, int b)
{
  int __time = 1;
  while(b>0)
  {
     __time += 7;
    if (a>b)
    {
      __time += 13;
      a=a-b;
    }
    else
    {
      __time += 17;
      b=b-a;
    }
  }
  __time += 21;
  assert(__time<=70);
  return a;
}

int nondet_int();

int main()
{
  gcd(nondet_int(),nondet_int());
//  int a=15, b=4;
//  gcd(a,b);
//  printf("mdc: %d\n", gcd(a,b));
  return 0;
}
