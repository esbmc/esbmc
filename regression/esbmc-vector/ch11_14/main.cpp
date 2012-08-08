// resizing vector
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector;

  unsigned int i;

  // set some initial content:
  for (i=1;i<10;i++) myvector.push_back(i);

  myvector.resize(5);
  myvector.resize(8,100);
  myvector.resize(12);

  cout << "myvector contains:";
  for (i=0;i<myvector.size();i++)
    cout << " " << myvector[i];

  cout << endl;

  return 0;
}
