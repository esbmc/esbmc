#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main()
{
	QVector<int> myints;
	cout << "0. size: " << (int) myints.size() << endl;

	for (int i=0; i<3; i++) myints.push_back(i);
		cout << "1. size: " << (int) myints.size() << endl;

	assert(myints.size() == 1000);

	return 0;
}
