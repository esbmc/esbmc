#include <iostream>
#include <cassert>
#include <QSet>
using namespace std;

int main ()
{
  QSet<int> first;
  first.insert(1);
  first.insert(2);
  first.insert(2);
  QSet<int> second;
  second.insert(1);
  second.insert(2);
  second.insert(3);
  QSet<int>::iterator it;

  assert(first.size() == 2);
  assert(second.size() == 3);

  first.swap(second);

  assert(first.size() == 3);
  assert(second.size() == 2);

  cout << "first contains:";
  for (it=first.begin(); it!=first.end(); it++) cout << " " << *it;

  cout << "\nsecond contains:";
  for (it=second.begin(); it!=second.end(); it++) cout << " " << *it;

  cout << endl;

  return 0;
}
