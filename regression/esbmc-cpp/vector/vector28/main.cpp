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
  assert(myvector.capacity() == 16); // WRONG: Implementation-dependent!
  cout << "max_size: " << myvector.max_size() << "\n";
  return 0;
}
