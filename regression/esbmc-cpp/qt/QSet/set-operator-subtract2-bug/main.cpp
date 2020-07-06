// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> first;
    first.insert(1);
    first.insert(2);
    first -= 1;
    assert(first.size() != 1);
    return 0;
}
