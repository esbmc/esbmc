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
  assert(vector.begin() == 2200000.22f);
}
