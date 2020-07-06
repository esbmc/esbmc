// QLinkedList::begin/end
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<int> myQLinkedList;
    myQLinkedList.push_back(10);
    myQLinkedList.push_back(13);
    QLinkedList<int>::const_iterator it;
    it = myQLinkedList.cend();
    it--;
    assert(*it != 13);
    
    cout << endl;
    
    return 0;
}
