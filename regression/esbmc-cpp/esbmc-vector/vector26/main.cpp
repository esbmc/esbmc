// vector::rbegin/rend
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;
  for (int i=1; i<=5; i++) myvector.push_back(i);

  cout << "myvector contains:";
  vector<int>::reverse_iterator rit;
  rit = myvector.rend();
  assert(*rit==0);
  for ( rit=myvector.rbegin() ; rit < myvector.rend(); ++rit )
    cout << " " << *rit;

  cout << endl;

  return 0;
}
