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

	first.append("o");
	first.append("u");

    assert(first.size() != 5);
  return 0;
}
