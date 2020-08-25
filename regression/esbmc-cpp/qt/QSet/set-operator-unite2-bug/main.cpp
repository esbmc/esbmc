// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> first;
    assert(first.size() == 0);
    first.insert(1);
    first.insert(2);
    first += 3;
    assert(first.size() != 3);
    return 0;
}
