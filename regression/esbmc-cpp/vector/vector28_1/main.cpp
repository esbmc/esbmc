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
  cout << "max_size: " << myvector.max_size() << "\n";

  // CORRECT - test standard guarantees:
  assert(myvector.capacity() >= myvector.size());  // Capacity ≥ size
  assert(myvector.capacity() >= 10);               // Can hold at least 10 elements
  assert(myvector.size() == 10);                   // Actual content is correct

  return 0;
}
