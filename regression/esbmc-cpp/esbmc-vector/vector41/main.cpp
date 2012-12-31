// vector::front
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;

  myvector.push_back(78);
  myvector.push_back(16);

  // now front equals 78, and back 16

  myvector.front() -= myvector.back();

  assert(myvector.front() == 62);
  cout << "myvector.front() is now " << myvector.front() << endl;

  return 0;
}
