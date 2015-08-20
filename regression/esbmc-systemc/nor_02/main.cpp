#include <systemc.h>

SC_MODULE(nor_gate)
{
	sc_inout<bool> a;
	sc_inout<bool> b;
	sc_out<bool> c;


	void or_process(void)
	{
		c = a.read() || b.read();		
	}

	void nor_process(void)
	{
		or_process();
		c = !c.read();
	} 

	void test_process(void)
	{
		assert( ( a.read() || b.read() ) == c.read() );
	}

	SC_CTOR(nor_gate)
	{
		// do nothing
	}
};

int sc_main( int argc, char * argv[] )
{
	sc_signal<bool> s1;
	sc_signal<bool> s2;
	sc_signal<bool> s3;

	s1.write(false);
	s2.write(true);
	s3.write(false);
	nor_gate gate("nor_gate");
	gate.a(s1);
	gate.b(s2);
	gate.c(s3);
	gate.nor_process();
	gate.test_process();

	return 0;
}
