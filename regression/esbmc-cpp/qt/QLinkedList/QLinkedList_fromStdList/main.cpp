// QLinkedList::back
#include <iostream>
#include <QLinkedList>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<int> myQLinkedList;
    list<int> mylist;
    
    myQLinkedList.push_back(10);
    mylist.push_back(20);
    mylist.push_back(20);
    myQLinkedList = myQLinkedList.fromStdList(mylist);
    assert(myQLinkedList.size() == 2);
    
    return 0;
}
