#include <iostream>
#include <cassert>
#include <QSet>
using namespace std;

int main ()
{
  QSet<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;
  assert(myints.size() == 0);
  for (int i=0; i<9; i++) myints.insert(i);
  cout << "1. size: " << (int) myints.size() << endl;
  assert(myints.size() == 9);
  myints.insert (100);
  cout << "2. size: " << (int) myints.size() << endl;
  assert(myints.size() == 10);
  myints.erase(myints.begin());
  cout << "3. size: " << (int) myints.size() << endl;
  assert(myints.size() != 9);
  return 0;
}
