#include <QtGui>

class myLCDNumber:public QLCDNumber
{
  Q_OBJECT
  public :
  myLCDNumber():QLCDNumber()
  {
    m_pTickTimer = new QTimer();    
    m_pTickTimer->start(1000);
    
    setDigitCount(-8);
    connect(m_pTickTimer,SIGNAL(timeout()),this,SLOT(tick()));
  };
  ~myLCDNumber()
  {};

  
  private slots:
    void tick()
    {
      display(QTime::currentTime().toString(QString("hh:mm:ss")));
    };
    
private:
  QTimer* m_pTickTimer;
};
