#include <vector>
namespace a = std;
class b;
class c
{
  virtual b *e();
};
template <class d>
class g
{
  a::vector<d> f;
};
class i : c
{
  g<i> h;
};
class b : c
{
  b *e();
};

int main()
{
  i j;
  return 0;
}