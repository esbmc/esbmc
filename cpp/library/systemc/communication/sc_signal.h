/***************************************************************
*     sc_signal.h
*
***************************************************************/

#ifndef SC_SIGNAL
#define SC_SIGNAL

template<class T>
class sc_signal
{

    public :
    
    sc_signal () { } // not implemented
  
    void write(T arg)
	{
		signal = arg;
	}

	T read()
	{
		return signal;
	}

    private: 
    	T signal;
    
};

#endif
