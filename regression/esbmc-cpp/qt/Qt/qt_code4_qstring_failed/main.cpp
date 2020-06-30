 #include <iostream>
 #include <QString>
#include<cassert>
#include<string>
 int main()
 {
     QString str("HelloWorld");
     assert(str.toStdString() != std::string("HelloWorld"));
     std::cout<<str.toStdString();
     return 0;
 }
