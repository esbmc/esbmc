#include "1.h"

struct X { int a; };

typedef const _Bool inteiro;

int y;

int fun();
void fun1(int a, int b){ int j, y; }
int fun2() { return 2; }
int fun3(int x) { return x; }
int fun5();
int fun6(int a, float b);

float x;

int fun7(float x, float y) 
{
  _Bool b;
  if(b)
    { return x; }
  else
  {
    x = b;
    return y;
  }

  return 0;
}

int fun8(float x, int y) 
{
  int b = 1;

  while(b==0) { int a; }

  for (int i = 0; i < x; ++i) { int i; int x = i; } 

  int i;
  for ( ; i < x; ++i) { int i=1; break; } 
  for (int i = 0; ; ++i) { int i=2; break; } 
  for (int i = 0; i < x; ) { int i=3; break; } 
  do { int i=4; break; } while(1);

  { int b; 

  for (int i = 0; i < x; ++i) { int i; int x = i; } 

  int i;
  for ( ; i < x; ++i) { int i=1; break; } 
  for (int i = 0; ; ++i) { int i=2; break; } 
  for (int i = 0; i < x; ) { int i=3; break; } 
  do { int i=4; break; } while(1);
  }

  return 0;
}

int fun9()
{
  return 0;
}

int main()
{
  typedef float ib;

  ib x1;
  float a =3;
  int x,k;
  x = x1;
  y = x;
  x = fun5();
  x = fun8(x,-4);

  { int x=3; }

}
