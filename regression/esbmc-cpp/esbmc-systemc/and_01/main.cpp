#include <systemc.h>
#include <cassert>

SC_MODULE(and_gatet)
{
	sc_inout<bool>  a;
	sc_inout<bool>  b;
	sc_out<bool> c;

	sc_clock clk;


	void and_process()
	{
	  c = a.read() && b.read();
	}

	void test_process()
	{
		while(true)
		{
			assert((a.read() & b.read()) == c.read());
			//wait();
			b.write(b.read());
		}
	}

	SC_CTOR(and_gatet)
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
	sc_signal<bool> s3;

	s1.write(true);
	s2.write(true);
	s3.write(false);
	
	and_gatet and_gate("and_gate");
	and_gate.a(s1);
	and_gate.b(s2);
	and_gate.c(s3);
 	
	and_gate.and_process();
	and_gate.test_process();

//	sc_start(100); 
	return 0;
}
