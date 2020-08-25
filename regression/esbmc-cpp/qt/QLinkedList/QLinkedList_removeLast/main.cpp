#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> first;
	first.push_back("a");

	first.push_back("e");

	first.push_back("i");
	
	first.removeLast();

    assert(first.size() == 2);
  return 0;
}
