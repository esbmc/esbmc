// QList::push_back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";

  do {
    cin >> myint;
    myQList.push_back (myint);
  } while (myint);

  cout << "myQList stores " << (int) myQList.size() << " numbers.\n";

  return 0;
}
