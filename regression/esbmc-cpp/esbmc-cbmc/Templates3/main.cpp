#include<cassert>

template<unsigned int size>
class int_array
{
public:
  int array[size];

  int read(unsigned int x)
  {
    assert(x<size);
    return array[x];
  }
};

int main()
{
  int_array<3> a;
  a.array[2]=1;

  assert(a.read(2)==1);
}
