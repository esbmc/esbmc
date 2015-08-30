// vector::operator[]
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector (10);   // 10 zero-initialized elements
  unsigned int i;

  vector<int>::size_type sz = myvector.size();

  // assign some values:
  for (i=0; i<sz; i++) myvector[i]=i;

  // reverse vector using operator[]:
  for (i=0; i<sz/2; i++)
  {
    int temp;
    temp = myvector[sz-1-i];
    myvector[sz-1-i]=myvector[i];
    myvector[i]=temp;
  }

  cout << "myvector contains:";
  for (i=0; i<sz; i++)
    cout << " " << myvector[i];

  cout << endl;

  return 0;
}
