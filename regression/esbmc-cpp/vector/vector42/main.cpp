// vector::back
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;

  myvector.push_back(10);

  while (myvector.back() != 0)
  {
    myvector.push_back ( myvector.back() -1 );
  }
  assert(myvector.back() == 0);
  cout << "myvector contains:";
  for (unsigned i=0; i<myvector.size() ; i++)
    cout << " " << myvector[i];

  cout << endl;

  return 0;
}
