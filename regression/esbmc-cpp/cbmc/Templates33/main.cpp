#include<cassert>

template<typename T>
class X
{
public:
  typename T::asd asd;
};

typedef X<char> Z;

int main()
{
  Z a;
  return 0;
}
