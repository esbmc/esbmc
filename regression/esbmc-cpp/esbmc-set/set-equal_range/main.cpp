#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  set<int> myset;
  pair<set<int>::iterator,set<int>::iterator> ret;

  for (int i=1; i<=5; i++) myset.insert(i*10);   // set: 10 20 30 40 50
  ret = myset.equal_range(30);
  assert(*ret.first == 	30);
  assert(*ret.second == 	40);
  cout << "lower bound points to: " << *ret.first << endl;
  cout << "upper bound points to: " << *ret.second << endl;

  return 0;
}
