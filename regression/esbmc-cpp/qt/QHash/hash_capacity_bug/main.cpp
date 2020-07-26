#include <cassert>
#include <QHash>
#include <iostream>

using namespace std;

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: iterator it;

    for(int i = 0; i < 9; i++)
    {
        myQHash[i] = i;
    }

    // show content:
    for ( it=myQHash.begin() ; it != myQHash.end(); it++ )
      cout << it.key() << " => " << it.value() << endl;


    assert(myQHash.capacity() == 0);
    return 0;
}
