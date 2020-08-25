/**
 *  CountDown Dialog With QT QLCDNumber
 *  Count down from 60 seconds to 0 Using QTimer
 *  as timer. 
 *  Change Color of QLCDNumber dynamically 
 *  when QCombobox Selection Changes.
 */
//myWindow.h file

#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QtGui>
 
class MyWindow: public QMainWindow
{
  Q_OBJECT
  
public:
    
	MyWindow():
	number(60)
	{};
	~ MyWindow(){};
	
	 void decorate(MyWindow *window)
	{	          	      
	      QWidget *centralWidget = new QWidget(window);
	      timer =new QTimer(this);
	      palette = new QPalette();
	      
	      QVBoxLayout* layout = new QVBoxLayout(centralWidget);
	            
	      LCD = new QLCDNumber();
	      comboBox = new QComboBox();

	      comboBox->addItem("BLACK");
	      comboBox->addItem("RED");
	      comboBox->addItem("GREEN");
	      comboBox->addItem("BLUE");
	      
	      
	      layout->addWidget(LCD);
	      layout->addWidget(comboBox);      	      
	      
	      
	      QObject::connect(timer ,  SIGNAL(timeout ()),this,SLOT(setNumber()));
	      QObject::connect(comboBox ,  SIGNAL(currentIndexChanged (int)),this,SLOT(setLCDColor(int)));
	      
	      LCD->setDigitCount(2);
	      LCD->setFixedSize(300, 300);
	      
	      window->setCentralWidget(centralWidget);
	      window->show();


	       //Set QTimer timeout to 1000 milliseconds (Update Number display each second)
	      
	      timer->start(1000);
	      LCD->display(number--);
	  
	};
	
	
public slots:    
  
	void setNumber()
	{
	  if(number < 0)
	  {
	    timer->stop();
	  }  
	  else
	  {  
	      LCD->display(number--);  
	  }   
	};
	
	void setLCDColor(int index)
	{

	    //Set QLCDNumber Color Here
	    if(index == 0)
	    {
	      palette->setColor(QPalette::WindowText,Qt::black);
	    } 
	    else if(index == 1)
	    {
	      palette->setColor(QPalette::WindowText,Qt::red);
	    }  
	    else if(index == 2)
	    {
	      palette->setColor(QPalette::WindowText,Qt::green);
	    }
	    else
	    {
	      palette->setColor(QPalette::WindowText,Qt::blue);
	    } 
	    LCD->setPalette(*palette);
	  
	};
	
public:
      
    QLCDNumber *LCD;
    QTimer *timer;
    QComboBox* comboBox;
    QPalette* palette;
    int number;
  
};
 #endif 
