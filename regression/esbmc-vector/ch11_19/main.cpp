// vector::at
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector (10);   // 10 zero-initialized ints
  unsigned int i;

  // assign some values:
  for (i=0; i<myvector.size(); i++)
    myvector.at(i)=i;

  cout << "myvector contains:";
  for (i=0; i<myvector.size(); i++)
    cout << " " << myvector.at(i);

  cout << endl;

  return 0;
}
