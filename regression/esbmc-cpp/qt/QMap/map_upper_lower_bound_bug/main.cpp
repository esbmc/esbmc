// QMap::lower_bound/upper_bound
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
    QMap<char,int> myQMap;
    QMap<char,int>::iterator it,itlow,itup;
    
    myQMap['a']=20;
    myQMap['b']=40;
    myQMap['c']=60;
    myQMap['d']=80;
    myQMap['e']=100;
    
    itlow=myQMap.lowerBound ('b');  // itlow points to b
    assert(itlow.key() != 'b');
    itup=myQMap.upperBound ('d');   // itup points to e (not d!)
    assert(itup.key() != 'e');
    myQMap.erase(itlow);        // erases [itlow,itup)
    
    // print content:
    for ( it=myQMap.begin() ; it != myQMap.end(); it++ )
        cout << it.key() << " => " << it.value() << endl;
    assert(myQMap.size() == 4);
    
    return 0;
}
