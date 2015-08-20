// list::size
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;
  assert(myints.size() == 0);
  for (int i=0; i<10; i++) myints.push_back(i);
  cout << "1. size: " << (int) myints.size() << endl;
  assert(myints.size() == 10);
  myints.insert (myints.begin(),10,100);
  cout << "2. size: " << (int) myints.size() << endl;
  assert(myints.size() != 20);
  myints.pop_back();
  cout << "3. size: " << (int) myints.size() << endl;
  assert(myints.size() != 19);
  return 0;
}
