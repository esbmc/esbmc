
/* structs used as input/output to threads */
struct inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv{
    int input_0;
    int input_1;
    int output_0;
};
struct inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv my_inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv;



struct inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv{
    int input_0;
    int input_1;
    int output_0;
};
struct inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv my_inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv;



struct inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv{
    int input_0;
    int input_1;
    int output_0;
};
struct inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv my_inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv;



struct inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv{
    int input_0;
    int input_1;
    int output_0;
};
struct inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv;



struct inputs_DUT{
    int input_0;
    int input_1;
    int output_0;
};
struct inputs_DUT my_inputs_DUT;




/* global join variables used to signal thread completion instead of pthread_join */
_Bool DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv_join = 0;
_Bool DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv_join = 0;
_Bool DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv_join = 0;
_Bool DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv_join = 0;
_Bool DUT_join = 0;

/* global declaration of threads */
void *DUT(void *arg);
