//LCDNumber.h File

#ifndef LCDNUMBER_H
#define LCDNUMBER_H

//#include <QLCDNumber>
#include <QtGui>
//#include <QTimer>
//#include <QTime>
class LCDNumber: public QLCDNumber
{
  Q_OBJECT  
	
  public:
  QTimer* timer;
  QTime*  timeValue;
  
  public:    
	LCDNumber(QWidget * parentWidget,int minutes,int seconds)
	{
	    timer = new QTimer();
	    timeValue = new QTime(0,minutes,seconds);
	    this->setParent(parentWidget);
	    this->display(timeValue->toString());
	    QObject::connect(timer,SIGNAL(timeout()),this,SLOT(setDisplay()));
	};
	~ LCDNumber(){};
	
   public slots:    
	void setDisplay()
	{
	  this->timeValue->setHMS(0,this->timeValue->addSecs(-1).minute(),this->timeValue->addSecs(-1).second());
	  this->display(this->timeValue->toString());	
	};
};
#endif
