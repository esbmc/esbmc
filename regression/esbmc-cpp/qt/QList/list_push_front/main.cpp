// QList::push_front
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;         // two ints with a value of 100
  myQList.push_back(100);
  myQList.push_back(100);
  assert(myQList.front() == 100);
  myQList.push_front (200);
  assert(myQList.front() == 200);
  myQList.push_front (300);
  assert(myQList.front() == 300);

  cout << "myQList contains:";
  for (QList<int>::iterator it=myQList.begin(); it!=myQList.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
