// QList::back
#include <iostream>
#include <QList>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
    QList<int> myQList;
    QSet<int> myQSet;
    
    myQList.push_back(10);
    myQList = myQList.fromSet(myQSet);
    assert(myQList.size() == 0);
    
    return 0;
}
