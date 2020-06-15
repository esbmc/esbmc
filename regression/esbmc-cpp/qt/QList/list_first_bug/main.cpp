// QList::front
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;

  myQList.push_back(77);
  myQList.push_back(22);
  assert(myQList.first() == 77);
  // now front equals 77, and back 22

  myQList.first() -= myQList.back();
  assert(myQList.first() != 55);
  cout << "myQList.front() is now " << myQList.first() << endl;

  return 0;
}
