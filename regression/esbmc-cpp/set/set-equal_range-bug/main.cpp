#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[5] = {10,20,30,40,50};
  set<int> myset(b,b+5);
  pair<set<int>::iterator,set<int>::iterator> ret;

  ret = myset.equal_range(30);
  assert(*ret.first == 	20);
//  assert(*ret.second !=	40);

    cout << "lower bound points to: " << *ret.first << endl;
  cout << "upper bound points to: " << *ret.second << endl;

  return 0;
}
