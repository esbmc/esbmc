//LCDNumber.h File

#ifndef LCDNUMBER_H
#define LCDNUMBER_H

#include <QtGui>
class LCDNumber: public QLCDNumber
{
  Q_OBJECT  
	
  public:
  QTimer* timer;
  QTime*  timeValue;
  QPushButton* button1;
  QPushButton* button2;
  
  public:    
	LCDNumber(QVBoxLayout*& verticalLayout,QMainWindow * window,int minutes,int seconds)
	{
	    timer	= 	new QTimer();
	    timeValue	=	new QTime(0,minutes,seconds,0);
	    button1	=	new QPushButton("Start");
	    button2	=	new QPushButton("Stop");
	    button1->setAutoExclusive(true);
	    button2->setAutoExclusive(true);
	    button1->setCheckable(true);
	    button2->setCheckable(true);
	    
	    
	    
	    verticalLayout->addWidget(this);
	    verticalLayout->addWidget(button1);
	    verticalLayout->addWidget(button2);
	    
	    
	    this->display(timeValue->toString(QString("mm:ss.zzz")));
	    QObject::connect(timer,SIGNAL(timeout()),this,SLOT(setDisplay()));
	    QObject::connect(button1,SIGNAL(pressed()),this,SLOT(start()));
	    QObject::connect(button2,SIGNAL(pressed()),this,SLOT(stop()));
	};
	~ LCDNumber(){};
	
   public slots:    
	void setDisplay()
	{
	  this->timeValue->setHMS(0,this->timeValue->addMSecs(-1).minute(),
				    this->timeValue->addMSecs(-1).second(),
				    this->timeValue->addMSecs(-1).msec());
	  this->display(this->timeValue->toString(QString("mm:ss.zzz")));
	};
	
	void start(){
	  this->timer->start();};
	
	void stop(){
	  this->timer->stop();};
	
};
#endif
