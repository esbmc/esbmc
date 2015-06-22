#ifndef SC_SIGNAL_PORTS_H
#define SC_SIGNAL_PORTS_H

#include "signal.h"
template<class T> class sc_signal; // pre-declaration of sc_signal

#define sc_out sc_inout

template <class T>
class sc_inout
{
	public :
    sc_inout() {};//not implemented

    void write(T arg)
	{
		signal.write( arg );
	}

	T read()
	{
		return signal.read();
	}

 	void operator()(sc_signal<T>& flag)
    {
		signal.write( flag.read() );
	}

	void operator= ( T flag )
	{
		this->signal.write(flag);	
	}
	
	private :
	sc_signal<T> signal;
};

#endif
