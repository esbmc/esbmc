// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> second;
    second.insert(1);
    second.insert(2);
    second |= 3;
    assert(second.size() != 3);
    return 0;
}
