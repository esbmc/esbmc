#include <cassert>

template < class T >
T teste(T value1, T value2, T value3)
{
   return value1;
}

template < >
int teste(int value1, int value2, int value3)
{
   return value2;
}

template < >
double teste(double value1, double value2, double value3)
{
   return value3;
}

int main()
{
  float z = teste(1.0f, 2.0f, 3.0f);
  assert(z==1.0f);

  double y = teste(1.0, 2.0, 3.0);
  assert(y==3.0);

  int x = teste(1, 2, 3);
  assert(x==2);
}

