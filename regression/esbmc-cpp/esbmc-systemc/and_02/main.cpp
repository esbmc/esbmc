// and gate in systemc
// @author Luciano Sobral <sobral.luciano@gmail.com>


#include <systemc.h>

SC_MODULE( and_gate )
{
	sc_inout<bool> a;
	sc_inout<bool> b;
	sc_out<bool> c;	

	SC_CTOR( and_gate )
	{
		// while function sc_start() still incomplete, the constructor
		// does nothing
	}


	// waiting while concurrency is not implemented
	void and_process( void )
	{		
		c = a.read() || b.read(); // wrong operation for an and_gate
	} 
	
	void and_test( void )
	{
		//should not pass for a != b
		assert( c.read() ==  ( a.read() && b.read() ) ); 
	}
		
};

int sc_main( int argc, char * argv[] )
{
	sc_signal<bool> s1;
	sc_signal<bool> s2;
	sc_signal<bool> s3;
	
	s1.write(false);
	s2.write(true);
	s3.write(true);	

	and_gate sample("and_gate");

	sample.a(s1);
	sample.b(s2);
	sample.c(s3);

	sample.and_test();// it fails, a = false and b = true  but c should be false
		
	return 0;
}
