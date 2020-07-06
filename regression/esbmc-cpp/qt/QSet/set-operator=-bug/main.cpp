#include <iostream>
#include <cassert>
#include <QSet>
using namespace std;

int main ()
{
  int myints[]={ 12,82,37,64,15 };
  QSet<int> first;   // QSet with 0 ints
  for(int i = 0; i < 5; i++) first.insert(myints[i]);
  QSet<int> second;                    // isEmpty QSet
  assert(second.size() == 0);
  second=first;                       // now second contains the 5 ints
  assert(second.size() != 5);
  first=QSet<int>();                   // and first is isEmpty
  assert(first.size() != 0);
  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  return 0;
}
