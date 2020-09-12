#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <systemc.h>
#include <cassert>
#include <stdio.h>

using namespace std;

SC_MODULE(mux){
	
	sc_inout<int> entry1;
	sc_inout<int> entry2;
	sc_inout<int> entry3;
	sc_inout<int> entry4;
	sc_inout<int> control1;
	sc_inout<int> control2;
	sc_out<int> out;
	
	void do_mux(){
		cout<<"entry1 "<<entry1.read()<<endl;
		cout<<"entry2 "<<entry2.read()<<endl;
		cout<<"entry3 "<<entry3.read()<<endl;
		cout<<"entry4 "<<entry4.read()<<endl;
		cout<<"control1 "<<control1.read()<<endl;
		cout<<"control2 "<<control2.read()<<endl<<endl;
		
		if((entry1.read()==1) && (control1.read()==5) && (control2.read()==5)) out.write(entry1.read());
		
		else if((entry2.read()==2) && (control1.read())==5 && !(control2.read()==5)) out.write(entry2.read());
		
		else if((entry3.read()==3) && !(control1.read()==5) && (control2.read()==5)) out.write(entry3.read());
			
		else if((entry4.read()==4) && !(control1.read()) && !(control2.read()==5)) out.write(entry4.read());
		
		cout<< "out " << out.read() << endl<<endl;
		
		
		
		
		

	}
	
	void test(){
		while(1){
			
	do_mux();
	
		control1.write(5);
		control2.write(0);
		
		wait(10, SC_NS);
		
		do_mux();
		
		control1.write(0);
		control2.write(5);
		
		wait(10, SC_NS);
		
		do_mux();
		
		control1.write(5);
		control2.write(5);
		
		wait(10, SC_NS);
		
		do_mux();
		
		control1.write(0);
		control2.write(0);
		
		wait(10, SC_NS);
	
	
}
}
	
	SC_CTOR(mux){
		SC_THREAD(test);
		sensitive <<entry1 << entry2 << entry3 << entry4 << control1 << control2;
		
		
	
	}
};


int sc_main( int argc, char* argv[] )
{
	
	sc_signal<int> in1;
	sc_signal<int> in2;
	sc_signal<int> in3;
	sc_signal<int> in4;
	sc_signal<int> ctrl1;
	sc_signal<int> ctrl2;
	sc_signal<int> out;
	
	in1.write(1);
	in2.write(2);
	in3.write(3);
	in4.write(4);
	ctrl1.write(0);
	ctrl2.write(0);
	out.write(5);
	
	mux mux("mux");
 	mux.entry1(in1);
	mux.entry2(in2);
	mux.entry3(in3);
	mux.entry4(in4);
	mux.control1(ctrl1);
	mux.control2(ctrl2);
	mux.out(out);
	
 	
	
	
	

	

	sc_start(); 
	return 0;
}

