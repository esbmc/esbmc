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

    first.insert(first.end(), "o");
    
    first.insert(first.end(), "u");

    assert(first.size() != 5);
  return 0;
}
