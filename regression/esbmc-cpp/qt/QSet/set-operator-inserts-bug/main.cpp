// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> first;
    assert(first.size() == 0);
    first << 1 << 2;
    assert(first.size() != 2);
    return 0;
}
