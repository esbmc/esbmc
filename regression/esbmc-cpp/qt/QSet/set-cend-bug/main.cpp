#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  QSet<int> myQSet;

  for(int i = 0; i < 5; i++) myQSet.insert(myints[i]);


  QSet<int>::const_iterator it = myQSet.cend();

  it--;
  cout << "myQSet.end(): " << *(it) << endl;

 
  for(int i = 0; i < 5; i++)
    if(*(it) == myints[i])
	  assert(false);

  return 0;
}
