 #include <iostream>
 #include <QString>

 int main()
 {
	  const char* tmp = NULL;
     QString str(tmp);
     std::cout<<str.toStdString();
     return 0;
 }
