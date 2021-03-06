// QVector::begin/end
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
    int myints[] = {75,23,65,42,13};
    QVector<int> myQVector;
    
    for(int i = 0; i < 5; i++)
    {
        myQVector.push_back(myints[i]);
    }
    
    QVector<int>::iterator it;
    it = myQVector.end();
    it--;
    assert(*it != 13);
    
    cout << endl;
    
    return 0;
}
