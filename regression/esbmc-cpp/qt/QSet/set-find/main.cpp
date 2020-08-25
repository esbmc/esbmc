#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  QSet<int>::iterator it;
  int i;
  assert(myQSet.size() == 0);
  // QSet some initial values:
  for (i=1; i<=5; i++) myQSet.insert(i*10);    // QSet: 10 20 30 40 50
  assert(myQSet.size() == 5);
  i = 10;
  for (it = myQSet.begin(); it != myQSet.end(); it++)
  {
    assert(*it == i);
    i+=10;
   }
  it=myQSet.find(20);
  assert(*it == 20);
  myQSet.erase (it);
  myQSet.erase (myQSet.find(40));
  it = myQSet.begin();
  it++;it++;
  assert(*it == 50);

  cout << "myQSet contains:";
  for (it=myQSet.begin(); it!=myQSet.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
