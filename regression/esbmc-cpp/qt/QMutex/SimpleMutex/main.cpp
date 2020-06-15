#include <QtCore>

QMutex mutex;
int number = 6;


class ClassA : public QThread
{
public:

    void run() Q_DECL_OVERRIDE
    {
        mutex.lock();
        number *= 5;
        number /= 4;
        mutex.unlock();
    }
};

class ClassB : public QThread
{
public:

    void run() Q_DECL_OVERRIDE
    {
        mutex.lock();
        number *= 3;
        number /= 2;
        mutex.unlock();
    }

};


int main()
{
    ClassA method1;
    ClassB method2;
    method1.start();
    method2.start();
    method1.wait();
    method2.wait();
    return 0;
}
