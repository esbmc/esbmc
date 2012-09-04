// resizing vector
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;

  unsigned int i;

  // set some initial content:
  for (i=1;i<10;i++) myvector.push_back(i);

  myvector.resize(5);
  assert(myvector.size() == 5);
  myvector.resize(8,100);
  assert(myvector.size() == 8);
  myvector.resize(12);
  assert(myvector.size() == 12);

  cout << "myvector contains:";
  for (i=0;i<myvector.size();i++)
    cout << " " << myvector[i];

  cout << endl;

  return 0;
}
