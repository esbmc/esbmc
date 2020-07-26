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
    first.insert(1);
    first.insert(1);
    first.insert(2);
    first &= 1;
    assert(first.size() == 1);
    return 0;
}
