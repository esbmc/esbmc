//#include <assert.h>
//#include <stdio.h>

#define TRUE 1
#define FALSE 0
_Bool in8;
_Bool in9;
_Bool in10;
_Bool in11;
_Bool in12;
_Bool in13;
_Bool in18=1;
_Bool Model_Outputs3=0;
_Bool Model_Outputs4=0;
int counter_a_DSTATE;
int counter_b_DSTATE;
int Unit_Delay1_d_DSTATE;
int Unit_Delay1_f_DSTATE;
int Unit_Delay1_h_DSTATE;
int Unit_Delay1_j_DSTATE;
int Unit_Delay1_b_DSTATE;
_Bool Unit_Delay_c_DSTATE;
_Bool Unit_Delay_e_DSTATE;
_Bool Unit_Delay_h_DSTATE;
_Bool Unit_Delay3_c_DSTATE;
_Bool Unit_Delay_i_DSTATE;
_Bool Unit_Delay_j_DSTATE;
_Bool Unit_Delay_a_DSTATE;
_Bool Unit_Delay_b_DSTATE;
_Bool Unit_Delay3_a_DSTATE;
_Bool Unit_Delay_d_DSTATE;
_Bool Unit_Delay_f_DSTATE;
_Bool Unit_Delay3_b_DSTATE;
_Bool Unit_Delay3_d_DSTATE;
_Bool Unit_Delay3_e_DSTATE;
_Bool Unit_Delay1_a_DSTATE;
_Bool Unit_Delay1_c_DSTATE;
_Bool Unit_Delay1_e_DSTATE;
_Bool Unit_Delay1_g_DSTATE;
_Bool Unit_Delay1_i_DSTATE;
_Bool Unit_Delay_g_DSTATE;
int Switch7;
int Switch1_m;
int add_g;
int Switch1_k;
int add_f;
int Switch1_i;
int add_e;
int Switch_e;
int add_d;
int Switch_d;
int add_c;
int Switch1_e;
int add_b;
int Switch1_c;
int add_a;
_Bool Data_Type_Conversion_a;
_Bool Switch5;
_Bool Switch4_a;
_Bool OR;
_Bool Switch6;
_Bool Switch2_a;
_Bool Switch3_a;
_Bool superior_e;
_Bool Switch3_f;
_Bool Switch1_l;
_Bool Switch2_f;
_Bool Switch4_f;
_Bool and1_d;
_Bool superior_d;
_Bool Switch3_e;
_Bool Switch1_j;
_Bool Switch2_e;
_Bool Switch4_e;
_Bool and1_c;
_Bool Switch_g;
_Bool Logical_Operator2;
_Bool superior_c;
_Bool Switch3_d;
_Bool Switch1_h;
_Bool Switch2_d;
_Bool Switch4_d;
_Bool and1_b;
_Bool Data_Type_Conversion_d;
_Bool Data_Type_Conversion_c;
_Bool Unit_Delay_g;
_Bool Multiport_Switch;
_Bool and_e;
_Bool superior_b;
_Bool Switch3_c;
_Bool Switch1_d;
_Bool Switch2_c;
_Bool Switch4_c;
_Bool and_d;
_Bool superior_a;
_Bool Switch3_b;
_Bool Switch1_b;
_Bool Switch2_b;
_Bool Switch4_b;
_Bool and1_a;
_Bool and_c;
_Bool and_b;
_Bool Switch1_a;
_Bool Switch_a;
_Bool Warning_Acti_ZCE;
int rtb_Switch_i;
int rtb_Switch_h;
int rtb_Switch_f;
int rtb_Switch1_g;
int rtb_Switch1_f;
int rtb_Switch_c;
int rtb_Switch_b;
_Bool rtb_and_a;
_Bool rtb_Logical_Operator;
_Bool rtb_Logical_Operator1;

int f1();

main(){
  // initializations
	Unit_Delay_g = FALSE;
	Unit_Delay_a_DSTATE = FALSE;
	Unit_Delay_g_DSTATE = FALSE;
	Unit_Delay_b_DSTATE = FALSE;
	Unit_Delay3_a_DSTATE = FALSE;
	Unit_Delay_c_DSTATE = FALSE;
	Unit_Delay1_b_DSTATE = 0;
	Unit_Delay1_a_DSTATE = FALSE;
	Unit_Delay_d_DSTATE = FALSE;
	Unit_Delay_e_DSTATE = FALSE;
	Unit_Delay1_d_DSTATE = 0;
	Unit_Delay1_c_DSTATE = FALSE;
	Unit_Delay_f_DSTATE = FALSE;
	counter_b_DSTATE = 0;
	Unit_Delay3_d_DSTATE = FALSE;
	Unit_Delay_i_DSTATE = FALSE;
	Unit_Delay1_h_DSTATE = 0;
	Unit_Delay1_g_DSTATE = FALSE;
	counter_a_DSTATE = 0;
	Unit_Delay3_e_DSTATE = FALSE;
	Unit_Delay_j_DSTATE = FALSE;
	Unit_Delay1_j_DSTATE = 0;
	Unit_Delay1_i_DSTATE = FALSE;
	Unit_Delay3_c_DSTATE = FALSE;
	Unit_Delay3_b_DSTATE = FALSE;
	Unit_Delay_h_DSTATE = FALSE;
	Unit_Delay1_f_DSTATE = 0;
	Unit_Delay1_e_DSTATE = FALSE;

	// verification loop

	// number of steps where the output has been consecutively true
	int count = 0;
	// number of verification steps
        int nbSteps = 10;
	for(int i=0; i<nbSteps;i++) {
		f1();
                // the output has been consecutively true one more step
		if (Model_Outputs4)
                   count++;
		else
		  // the output has not been consecutively true
                  count=0;
	}
        // lights should never remain lit (i.e. count must be less than nbSteps)
	assert (count<nbSteps); 

}

// non deterministic boolean value
_Bool nondet_in();

// function that describes the flasher module
f1() {
		in8=nondet_in();
		in9=nondet_in();
		in10=nondet_in();
		in11=nondet_in();
		in12=nondet_in();
		in13=nondet_in();



	if (in18) {
		Data_Type_Conversion_a = in11;
		and_b = ((in10 == TRUE) && (TRUE != Unit_Delay_a_DSTATE));
		Unit_Delay_g = Unit_Delay_g_DSTATE;
		switch (((1 + in12) + (Data_Type_Conversion_a * 2))) {
			case 1 : Multiport_Switch = Unit_Delay_g;
 				break;
			case 2 : Multiport_Switch = TRUE;
 				break;
			case 3 : Multiport_Switch = FALSE;
 				break;
		}
		and_c = ((Unit_Delay_g == FALSE) && (FALSE != Unit_Delay_b_DSTATE));
		if (in13) {
			Switch5 = in8;
		}
		else {
			Switch5 = FALSE;
		}
		and1_a = ((Switch5 == TRUE) && (TRUE != Unit_Delay3_a_DSTATE));
		if ((TRUE == ((and1_a - Unit_Delay_c_DSTATE) != 0))) {
			rtb_Switch_b = 0;
		}
		else {
			add_a = (1 + Unit_Delay1_b_DSTATE);
			rtb_Switch_b = add_a;
		}
		superior_a = (rtb_Switch_b >= 3);
		if (superior_a) {
			Switch1_c = 0;
		}
		else {
			Switch1_c = rtb_Switch_b;
		}
		if (Switch5) {
			if (and1_a) {
				Switch1_b = TRUE;
			}
			else {
				if (superior_a) {
					if (Unit_Delay1_a_DSTATE) {
						Switch4_b = FALSE;
					}
					else {
						Switch4_b = TRUE;
					}
					Switch2_b = Switch4_b;
				}
				else {
					Switch2_b = Unit_Delay1_a_DSTATE;
				}
				Switch1_b = Switch2_b;
			}
			Switch3_b = Switch1_b;
		}
		else {
			Switch3_b = FALSE;
		}
		if (in13) {
			Switch4_a = in9;
		}
		else {
			Switch4_a = FALSE;
		}
		and_d = ((Switch4_a == TRUE) && (TRUE != Unit_Delay_d_DSTATE));
		if ((1 == (and_d - Unit_Delay_e_DSTATE))) {
			rtb_Switch_c = 0;
		}
		else {
			add_b = (1 + Unit_Delay1_d_DSTATE);
			rtb_Switch_c = add_b;
		}
		superior_b = (rtb_Switch_c >= 3);
		if (superior_b) {
			Switch1_e = 0;
		}
		else {
			Switch1_e = rtb_Switch_c;
		}
		if (Switch4_a) {
			if (and_d) {
				Switch1_d = TRUE;
			}
			else {
				if (superior_b) {
					if (Unit_Delay1_c_DSTATE) {
						Switch4_c = FALSE;
					}
					else {
						Switch4_c = TRUE;
					}
					Switch2_c = Switch4_c;
				}
				else {
					Switch2_c = Unit_Delay1_c_DSTATE;
				}
				Switch1_d = Switch2_c;
			}
			Switch3_c = Switch1_d;
		}
		else {
			Switch3_c = FALSE;
		}
		rtb_and_a = (Data_Type_Conversion_a && ( ! Unit_Delay_g));
		OR = (rtb_and_a || Unit_Delay_g);
		and_e = ((OR == TRUE) && (TRUE != Unit_Delay_f_DSTATE));
		if (rtb_and_a) {
			Switch7 = 60;
		}
		else {
			Switch7 = 20;
		}
		if ((counter_b_DSTATE == 0)) {
			rtb_Switch1_g = 0;
		}
		else {
			rtb_Switch1_g = 1;
		}
		Data_Type_Conversion_d = (rtb_Switch1_g != 0);
		if (and_e) {
			Switch_e = Switch7;
		}
		else {
			add_d = (( - rtb_Switch1_g) + counter_b_DSTATE);
			Switch_e = add_d;
		}
		and1_c = ((Data_Type_Conversion_d == TRUE) && (TRUE != Unit_Delay3_d_DSTATE));
		if ((1 == (and1_c - Unit_Delay_i_DSTATE))) {
			rtb_Switch_h = 0;
		}
		else {
			add_f = (1 + Unit_Delay1_h_DSTATE);
			rtb_Switch_h = add_f;
		}
		superior_d = (rtb_Switch_h >= 1);
		if (superior_d) {
			Switch1_k = 0;
		}
		else {
			Switch1_k = rtb_Switch_h;
		}
		if (Data_Type_Conversion_d) {
			if (and1_c) {
				Switch1_j = TRUE;
			}
			else {
				if (superior_d) {
					if (Unit_Delay1_g_DSTATE) {
						Switch4_e = FALSE;
					}
					else {
						Switch4_e = TRUE;
					}
					Switch2_e = Switch4_e;
				}
				else {
					Switch2_e = Unit_Delay1_g_DSTATE;
				}
				Switch1_j = Switch2_e;
			}
			Switch3_e = Switch1_j;
		}
		else {
			Switch3_e = FALSE;
		}
		if ((counter_a_DSTATE == 0)) {
			rtb_Switch1_f = 0;
		}
		else {
			rtb_Switch1_f = 1;
		}
		Data_Type_Conversion_c = (rtb_Switch1_f != 0);
		if (and_c) {
			Switch_d = 10;
		}
		else {
			add_c = (( - rtb_Switch1_f) + counter_a_DSTATE);
			Switch_d = add_c;
		}
		and1_d = ((Data_Type_Conversion_c == TRUE) && (TRUE != Unit_Delay3_e_DSTATE));
		if ((1 == (and1_d - Unit_Delay_j_DSTATE))) {
			rtb_Switch_i = 0;
		}
		else {
			add_g = (1 + Unit_Delay1_j_DSTATE);
			rtb_Switch_i = add_g;
		}
		superior_e = (rtb_Switch_i >= 10);
		if (superior_e) {
			Switch1_m = 0;
		}
		else {
			Switch1_m = rtb_Switch_i;
		}
		if (Data_Type_Conversion_c) {
			if (and1_d) {
				Switch1_l = TRUE;
			}
			else {
				if (superior_e) {
					if (Unit_Delay1_i_DSTATE) {
						Switch4_f = FALSE;
					}
					else {
						Switch4_f = TRUE;
					}
					Switch2_f = Switch4_f;
				}
				else {
					Switch2_f = Unit_Delay1_i_DSTATE;
				}
				Switch1_l = Switch2_f;
			}
			Switch3_f = Switch1_l;
		}
		else {
			Switch3_f = FALSE;
		}
		rtb_Logical_Operator = (Switch3_e || Switch3_f);
		rtb_Logical_Operator1 = (Data_Type_Conversion_d || Data_Type_Conversion_c);
		if ((and_b && ( ! Warning_Acti_ZCE))) {
			if (in10) {
				Logical_Operator2 = ( ! Unit_Delay3_c_DSTATE);
				Switch_g = Logical_Operator2;
			}
			else {
				Switch_g = FALSE;
			}
			Unit_Delay3_c_DSTATE = Switch_g;
		}
		else {
		}
		Warning_Acti_ZCE = and_b;
		if (in13) {
			Switch6 = Switch_g;
		}
		else {
			Switch6 = FALSE;
		}
		and1_b = ((Switch6 == TRUE) && (TRUE != Unit_Delay3_b_DSTATE));
		if ((1 == (and1_b - Unit_Delay_h_DSTATE))) {
			rtb_Switch_f = 0;
		}
		else {
			add_e = (1 + Unit_Delay1_f_DSTATE);
			rtb_Switch_f = add_e;
		}
		superior_c = (rtb_Switch_f >= 3);
		if (superior_c) {
			Switch1_i = 0;
		}
		else {
			Switch1_i = rtb_Switch_f;
		}
		if (Switch6) {
			if (and1_b) {
				Switch1_h = TRUE;
			}
			else {
				if (superior_c) {
					if (Unit_Delay1_e_DSTATE) {
						Switch4_d = FALSE;
					}
					else {
						Switch4_d = TRUE;
					}
					Switch2_d = Switch4_d;
				}
				else {
					Switch2_d = Unit_Delay1_e_DSTATE;
				}
				Switch1_h = Switch2_d;
			}
			Switch3_d = Switch1_h;
		}
		else {
			Switch3_d = FALSE;
		}
		if (rtb_Logical_Operator1) {
			Switch2_a = rtb_Logical_Operator;
		}
		else {
			if (Switch6) {
				Switch1_a = Switch3_d;
			}
			else {
				Switch1_a = Switch3_b;
			}
			Switch2_a = Switch1_a;
		}
		if (rtb_Logical_Operator1) {
			Switch3_a = rtb_Logical_Operator;
		}
		else {
			if (Switch6) {
				Switch_a = Switch3_d;
			}
			else {
				Switch_a = Switch3_c;
			}
			Switch3_a = Switch_a;
		}
	}
	else {
	}
	Model_Outputs3 = Switch2_a;
	Model_Outputs4 = Switch3_a;
	if (in18) {
		Unit_Delay_a_DSTATE = in10;
		Unit_Delay_g_DSTATE = Multiport_Switch;
		Unit_Delay_b_DSTATE = Unit_Delay_g;
		Unit_Delay3_a_DSTATE = Switch5;
		Unit_Delay_c_DSTATE = and1_a;
		Unit_Delay1_b_DSTATE = Switch1_c;
		Unit_Delay1_a_DSTATE = Switch3_b;
		Unit_Delay_d_DSTATE = Switch4_a;
		Unit_Delay_e_DSTATE = and_d;
		Unit_Delay1_d_DSTATE = Switch1_e;
		Unit_Delay1_c_DSTATE = Switch3_c;
		Unit_Delay_f_DSTATE = OR;
		counter_b_DSTATE = Switch_e;
		Unit_Delay3_d_DSTATE = Data_Type_Conversion_d;
		Unit_Delay_i_DSTATE = and1_c;
		Unit_Delay1_h_DSTATE = Switch1_k;
		Unit_Delay1_g_DSTATE = Switch3_e;
		counter_a_DSTATE = Switch_d;
		Unit_Delay3_e_DSTATE = Data_Type_Conversion_c;
		Unit_Delay_j_DSTATE = and1_d;
		Unit_Delay1_j_DSTATE = Switch1_m;
		Unit_Delay1_i_DSTATE = Switch3_f;
		Unit_Delay3_b_DSTATE = Switch6;
		Unit_Delay_h_DSTATE = and1_b;
		Unit_Delay1_f_DSTATE = Switch1_i;
		Unit_Delay1_e_DSTATE = Switch3_d;
	}
	else {
	}
}
