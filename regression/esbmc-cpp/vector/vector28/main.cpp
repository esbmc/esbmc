// comparing size, capacity and max_size
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;

  // set some content in the vector:
  for (int i=0; i<10; i++) myvector.push_back(i);

  cout << "size: " << myvector.size() << "\n";
  cout << "capacity: " << myvector.capacity() << "\n";
  // capacity() after push_back is implementation-defined; this test exists
  // to demonstrate that ESBMC reports the assertion violation. See
  // vector28_1 for the portable form (capacity() >= size()).
  assert(myvector.capacity() == 16);
  cout << "max_size: " << myvector.max_size() << "\n";
  return 0;
}
