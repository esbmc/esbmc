//#include <QtCore>
#include <QList>
#include <QString>
#include <iostream>

int main(void)
{
    QList<QString>* list = new QList<QString>();
	 char* name = NULL;
    list->push_back(name);
    list->push_back("Element 2");
    list->push_back("Element 3");
    list->push_back("Element 4");
    
    
    list->indexOf("Element 3") == -1 ?
	   std::cout<<"No Element has been found!"<<std::endl
	  :std::cout<<"Element found at index "<<list->indexOf("Element 3")<<std::endl; 
}
