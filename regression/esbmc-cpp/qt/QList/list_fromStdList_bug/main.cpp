// QList::back
#include <iostream>
#include <QList>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
    QList<int> myQList;
    list<int> mylist;
    
    myQList.push_back(10);
    mylist.push_back(20);
    mylist.push_back(20);
    myQList = myQList.fromStdList(mylist);
    assert(myQList.size() == 1);
    
    return 0;
}
