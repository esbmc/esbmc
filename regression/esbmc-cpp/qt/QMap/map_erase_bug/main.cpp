// erasing from QMap
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
    QMap<char,int> myQMap;
    QMap<char,int>::iterator it;
    char chararray[5] = {'a', 'c', 'd', 'e', 'f'};
    int intarray[5] = {10, 30, 40, 50, 60};
    
    
    // insert some values:
    myQMap['a']=10;
    myQMap['b']=20;
    myQMap['c']=30;
    myQMap['d']=40;
    myQMap['e']=50;
    myQMap['f']=60;
    
    it=myQMap.find('b');
    myQMap.erase(it);                   // erasing by iterator
    
    int i = 0;
    // show content:
    for ( it=myQMap.begin() ; it != myQMap.end(); it++ ){
        cout << it.key() << " => " << it.value() << endl;
        assert(it.key() == chararray[i]);
        assert(it.value() != intarray[i]);
        i++;
    }
    
    return 0;
}
