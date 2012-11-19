#include <cassert>

template < class T >
T teste(T value1, T value2)
{
   return value1;
}

template < >
int teste(int value1, int value2)
{
   return value2;
}

int main()
{
  double y = teste(1.0, 2.0);
  assert(y==1.0);

  int x = teste(1,2);
  assert(x==2);
}

