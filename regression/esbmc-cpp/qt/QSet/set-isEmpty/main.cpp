#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;

  myQSet.insert(20);
  myQSet.insert(30);
  myQSet.insert(10);
  assert(myQSet.size() == 3);
  cout << "myQSet contains:";
  while (!myQSet.isEmpty())
  {
     cout << " " << *myQSet.begin();
     myQSet.erase(myQSet.begin());
  }
  assert(myQSet.begin() == myQSet.end());
  assert(myQSet.size() == 0);
  cout << endl;

  return 0;
}
