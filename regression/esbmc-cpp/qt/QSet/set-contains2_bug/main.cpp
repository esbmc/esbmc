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
    
    mySecondQSet.insert (100);
    mySecondQSet.insert (200);
    mySecondQSet.insert (300);
    
    assert(!(myQSet.contains(mySecondQSet)));
    
    return 0;
}
