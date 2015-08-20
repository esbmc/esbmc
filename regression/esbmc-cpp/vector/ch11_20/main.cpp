// vector::front
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector;

  myvector.push_back(78);
  myvector.push_back(16);

  // now front equals 78, and back 16

  myvector.front() -= myvector.back();

  cout << "myvector.front() is now " << myvector.front() << endl;

  return 0;
}
