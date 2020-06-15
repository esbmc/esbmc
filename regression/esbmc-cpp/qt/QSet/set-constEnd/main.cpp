#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  QSet<int> myQSet;

  for(int i = 0; i < 5; i++) myQSet.insert(myints[i]);


  QSet<int>::const_iterator it = myQSet.constEnd();

  it--;
  cout << "myQSet.end(): " << *(it) << endl;

 
  assert(*(it) == myints[0] || *(it) == myints[1] || *(it) == myints[2] || *(it) == myints[3] || *(it) == myints[4]);

  return 0;
}
