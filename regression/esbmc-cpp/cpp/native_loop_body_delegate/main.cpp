#include <cassert>
#include <vector>

// A loop whose body does not convert natively is delegated whole to
// convert_while / convert_for, after rolling back what the abandoned native
// attempt allocated. The container operational models produce that shape.
// Values and iteration counts must be unchanged by the delegation.
int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  int sum = 0;
  unsigned i = 0;
  while (i < v.size())
  {
    sum += v[i];
    ++i;
  }
  assert(sum == 6);

  int prod = 1;
  for (unsigned j = 0; j < v.size(); ++j)
    prod *= v[j];
  assert(prod == 6);

  return 0;
}
