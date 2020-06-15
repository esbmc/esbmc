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
    myQList = myQList.fromVector(myQVector);
    assert(myQList.size() != 0);
    
    return 0;
}
