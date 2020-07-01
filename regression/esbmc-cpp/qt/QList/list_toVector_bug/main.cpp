// QList::back
#include <iostream>
#include <QList>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
    QList<int> myQList;
    QVector<int> myQVector;
    
    myQList.push_back(10);
    myQVector = myQList.toVector();
    assert(myQVector.size() == 0);
    
    return 0;
}
