// QVector::size
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myints;
  cout << "0. length: " << (int) myints.length() << endl;
  assert(myints.length() == 0);
  for (int i=0; i<10; i++) myints.push_back(i);
  cout << "1. length: " << (int) myints.length() << endl;
  assert(myints.length() == 10);
  myints.insert (myints.begin(),10);
  cout << "2. length: " << (int) myints.length() << endl;
  assert(myints.length() == 11);
  myints.pop_back();
  cout << "3. length: " << (int) myints.length() << endl;
  assert(myints.length() == 10);
  return 0;
}
