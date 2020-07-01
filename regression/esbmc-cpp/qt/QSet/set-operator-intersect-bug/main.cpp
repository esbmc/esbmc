// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> first;
    assert(first.size() == 0);
    QSet<int> second;
    first.insert(1);
    first.insert(1);
    first.insert(1);
    first.insert(2);
    second.insert(1);
    first &= second;
    assert(first.size() != 1);
    return 0;
}
