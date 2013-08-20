#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  multiset<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;
  assert(myints.size() == 0);
  for (int i=0; i<10; i++) myints.insert(i);
  cout << "1. size: " << (int) myints.size() << endl;
  assert(myints.size() == 10);
  myints.insert (100);
  cout << "2. size: " << (int) myints.size() << endl;
  assert(myints.size() == 11);
  myints.erase(5);
  cout << "3. size: " << (int) myints.size() << endl;
  assert(myints.size() > 10);
  return 0;
}
