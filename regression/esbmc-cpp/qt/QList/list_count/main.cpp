// QList::size
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myints;
  cout << "0. count: " << (int) myints.count() << endl;
  assert(myints.count() == 0);
  for (int i=0; i<10; i++) myints.push_back(i);
  cout << "1. count: " << (int) myints.count() << endl;
  assert(myints.count() == 10);
  myints.insert (myints.begin(),10);
  cout << "2. count: " << (int) myints.count() << endl;
  assert(myints.count() == 11);
  myints.pop_back();
  cout << "3. count: " << (int) myints.count() << endl;
  assert(myints.count() == 10);
  return 0;
}
