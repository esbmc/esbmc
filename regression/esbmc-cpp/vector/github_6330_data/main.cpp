// github #6330: std::vector::data() ([vector.data], C++11) returns a pointer to
// the contiguous storage. It was missing from the model. data() and its const
// overload return the underlying buffer, so reads/writes alias the vector.
#include <cassert>
#include <vector>

int first(const std::vector<int> &v)
{
  return v.data()[0]; // const data()
}

int main()
{
  std::vector<int> v = {1, 2, 3};

  int *p = v.data();
  assert(p[0] == 1 && p[2] == 3);

  p[1] = 9; // write through data() aliases the vector
  assert(v[1] == 9);

  assert(first(v) == 1);
  return 0;
}
