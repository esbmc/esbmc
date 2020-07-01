#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  int b[5] = {10,20,30,40,50};
  QSet<int> myQSet;
  for(int i = 0; i < 5; i++) myQSet.insert(b[i]);
  QSet<int>::iterator it;
  int i;
  // QSet some initial values:
  assert(myQSet.size() == 5);
  i = 10;
  for (it = myQSet.begin(); it != myQSet.end(); it++){
    assert(*it == i);
    i += 10;
    }
  it=myQSet.find(20);
  assert(*it == 20);
  myQSet.erase (it);
  myQSet.erase (myQSet.find(40));
  it = myQSet.begin();
  it++;it++;
  assert(*it != 50);

  cout << "myQSet contains:";
  for (it=myQSet.begin(); it!=myQSet.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
