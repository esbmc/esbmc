#include <cassert>
template <typename T>
struct OneVector
{
  T element;
  OneVector(T e) : element(e)
  {
  }

  decltype(auto) begin()
  {
    return element;
  }
  T get_one()
  {
    return element;
  }
};
OneVector<float> vector(22.22f);
int main()
{
  assert(vector.get_one() == 22.22f);
}
