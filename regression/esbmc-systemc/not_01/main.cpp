#include <systemc.h>


SC_MODULE(not_gate)
{
	sc_inout<bool>  a;
	sc_out<bool>  b;

	sc_clock clk;


	void not_process(void)
	{
	  b = !a.read(); // b = not a
	}

	void test_process(void)
	{
			assert( a.read() == b.read() ); // fail 
	}

	SC_CTOR(not_gate)
	{

//		SC_METHOD(and_process);
//		sensitive << a;
//		sensitive << b;

//		SC_CTHREAD(test_process,clk);

	} 
};

int sc_main( int argc, char* argv[] )
{
	sc_signal<bool> s1;
	sc_signal<bool> s2;

	s1.write(true);
	s2.write(true);
	
	not_gate gate("not_gate");
	gate.a(s1);
	gate.b(s2);

	gate.not_process();
	gate.test_process();

//	sc_start(100); 
	return 0;
}
