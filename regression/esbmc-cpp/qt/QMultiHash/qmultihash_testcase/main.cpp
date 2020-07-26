#include <QMultiHash>
#include <cassert>
#include <QString>
#include <iostream>

using namespace std;

int main ()
{
    QMultiHash<QString, int> hash1, hash2, hash3;
    
    hash1.insert("plenty", 100);
    hash1.insert("plenty", 2000);
    // hash1.size() == 2
    
    hash2.insert("plenty", 5000);
    // hash2.size() == 1
    return 0;
}