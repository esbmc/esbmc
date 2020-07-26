// QList::begin/end
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  QList<int> myQList;
  for (int i = 0; i < 5; i++)
    myQList.push_back(myints[i]);
  QList<int>::const_iterator it;
  it = myQList.cend();
  it--;
  assert(*it == 13);
  
  cout << endl;

  return 0;
}
