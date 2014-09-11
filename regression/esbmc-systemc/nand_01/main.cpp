// nand gate
// Luciano Sobral <sobral.luciano@gmail.com>
// expected: verification sucessfull

#include <systemc.h>

SC_MODULE(nand_gate)
{
	sc_inout<bool> a;
	sc_inout<bool> b;
	sc_out<bool> c;

	void nand_process(void)
	{
		and_process(); // c = a and b
		c = !c.read(); // c = not c
	} 

	void and_process ( void )
	{	
		c = a.read() && b.read();
	}
	
	void test_process(void)
	{
		assert( c.read() != ( a.read() && b.read() ) );
	}

	SC_CTOR(nand_gate)
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
	s3.write(false);
	nand_gate gate("nand_gate");
	gate.a(s1);
	gate.b(s2);
	gate.c(s3);
	gate.nand_process();
	gate.test_process();

	return 0;
}
