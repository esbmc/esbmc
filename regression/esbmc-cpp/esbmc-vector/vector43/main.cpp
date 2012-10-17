// vector::begin
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector;
  for (int i=1; i<=5; i++) myvector.push_back(i);

  vector<int>::iterator it;

  cout << "myvector contains:";
  it=myvector.begin();
  assert(*it != 1);
  for ( it=myvector.begin() ; it < myvector.end(); it++ )
    cout << " " << *it;

  cout << endl;

  return 0;
}
