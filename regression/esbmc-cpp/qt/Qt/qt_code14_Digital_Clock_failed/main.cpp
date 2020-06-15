#include "myLcdNumber.h"
int main(int argc,char ** argv)
{
  QApplication app(argc,argv);

  myLCDNumber* number = new myLCDNumber();
  number->setWindowTitle("QLCDnumber Digital Clock");
  number->setFixedSize(300,100);
  number->show();  
  
  return app.exec();
};
