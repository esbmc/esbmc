#include <systemc.h>


SC_MODULE( or_gate )
{
	
	sc_inout<bool> a;
	sc_inout<bool> b;
	sc_out<bool> c;
	
	void or_process( void )
	{
		c = a.read() || b.read();
	}

	void test_process( void )
	{

			assert( (a.read() || b.read() ) == c.read() );
	}


	SC_CTOR( or_gate )
	{
		
	}
};


int sc_main( int argc, char * argv[] )
{
	sc_signal<bool> s1;
	sc_signal<bool> s2;
	sc_signal<bool> s3;


	s1.write(true);
	s2.write(false);
	s3.write(true);

	or_gate gate("or_gate");
	
	gate.a(s1);
	gate.b(s2);
	gate.c(s3);

//	gate.or_process();
	gate.test_process();

	return 0;
}
