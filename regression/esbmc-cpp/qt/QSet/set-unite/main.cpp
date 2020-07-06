#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QSet<int> myQSet;
    QSet<int> mySecondQSet;
    
    QSet<int>::iterator it;
    
    myQSet.insert (100);
    myQSet.insert (200);
    myQSet.insert (300);
    
    mySecondQSet.insert (400);
    mySecondQSet.insert (500);
    mySecondQSet.insert (100);
    myQSet.unite(mySecondQSet);
    assert(myQSet.size() == 5);
    
    return 0;
}
