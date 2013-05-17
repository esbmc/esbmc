#ifndef SC_SENSITIVE_H
#define SC_SENSITIVE_H

#include "../communication/sc_signal_ports.h"


template <class T> class sc_inout; // pre-definition

class sc_sensitive
{

	public :

		void operator << ( sc_inout<T> inout )
		{
	
		}

	private :
		sc_sensitive() {}
		~sc_sensitive() {}

	

};

extern sc_sensitive sensitive;

#endif
