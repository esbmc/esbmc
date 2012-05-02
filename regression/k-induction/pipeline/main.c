/*int nondet(void)
{
  int x;
  {
    return (x);
  }
  }*/

void error(void) 
{ 

  {
  goto ERROR;
  ERROR: ;
  return;
}
}
int main_in1_val  ;
int main_in1_val_t  ;
int main_in1_ev  ;
int main_in1_req_up  ;
int main_in2_val  ;
int main_in2_val_t  ;
int main_in2_ev  ;
int main_in2_req_up  ;
int main_diff_val  ;
int main_diff_val_t  ;
int main_diff_ev  ;
int main_diff_req_up ;
int main_sum_val  ;
int main_sum_val_t  ;
int main_sum_ev  ;
int main_sum_req_up ;
int main_pres_val  ;
int main_pres_val_t  ;
int main_pres_ev  ;
int main_pres_req_up  ;
int main_dbl_val  ;
int main_dbl_val_t  ;
int main_dbl_ev  ;
int main_dbl_req_up  ;
int main_zero_val  ;
int main_zero_val_t  ;
int main_zero_ev  ;
int main_zero_req_up  ;
int main_clk_val  ;
int main_clk_val_t  ;
int main_clk_ev  ;
int main_clk_req_up  ;
int main_clk_pos_edge  ;
int main_clk_neg_edge  ;
int N_generate_st  ;
int N_generate_i  ;
int S1_addsub_st  ;
int S1_addsub_i  ;
int S2_presdbl_st  ;
int S2_presdbl_i  ;
int S3_zero_st  ;
int S3_zero_i  ;
int D_z  ;
int D_print_st  ;
int D_print_i  ;
void N_generate(void) 
{ int a ;
  int b ;

  {
  main_in1_val_t = a;
  main_in1_req_up = 1;
  main_in2_val_t = b;
  main_in2_req_up = 1;

  return;
}
}
void S1_addsub(void) 
{ int a ;
  int b ;

  {
  a = main_in1_val;
  b = main_in2_val;
  main_sum_val_t = a + b;
  main_sum_req_up = 1;
  main_diff_val_t = a - b;
  main_diff_req_up = 1;

  return;
}
}
void S2_presdbl(void) 
{ int a ;
  int b ;
  int c ;
  int d ;

  {
  a = main_sum_val;
  b = main_diff_val;
  main_pres_val_t = a;
  main_pres_req_up = 1;
  c = a + b;
  d = a - b;
  main_dbl_val_t = c + d;
  main_dbl_req_up = 1;

  return;
}
}
void S3_zero(void) 
{ int a ;
  int b ;

  {
  a = main_pres_val;
  b = main_dbl_val;
  main_zero_val_t = b - (a + a);
  main_zero_req_up = 1;

  return;
}
}
void D_print(void) 
{ 

  {
  D_z = main_zero_val;

  return;
}
}
void eval(void) 
{ int tmp ;
  int tmp___0 ;
  int tmp___1 ;
  int tmp___2 ;
  int tmp___3 ;

  {
  {
  while (1) {
    while_0_continue: /* CIL Label */ ;
    if ((int )N_generate_st == 0) {

    } else {
      if ((int )S1_addsub_st == 0) {

      } else {
        if ((int )S2_presdbl_st == 0) {

        } else {
          if ((int )S3_zero_st == 0) {

          } else {
            if ((int )D_print_st == 0) {

            } else {
              goto while_0_break;
            }
          }
        }
      }
    }
    if ((int )N_generate_st == 0) {
      {
	tmp = __VERIFIER_nondet_int();
      }
      if (tmp) {
        {
        N_generate_st = 1;
        N_generate();
        }
      } else {

      }
    } else {

    }
    if ((int )S1_addsub_st == 0) {
      {
	tmp___0 = __VERIFIER_nondet_int();
      }
      if (tmp___0) {
        {
        S1_addsub_st = 1;
        S1_addsub();
        }
      } else {

      }
    } else {

    }
    if ((int )S2_presdbl_st == 0) {
      {
	tmp___1 = __VERIFIER_nondet_int();
      }
      if (tmp___1) {
        {
        S2_presdbl_st = 1;
        S2_presdbl();
        }
      } else {

      }
    } else {

    }
    if ((int )S3_zero_st == 0) {
      {
	tmp___2 = __VERIFIER_nondet_int();
      }
      if (tmp___2) {
        {
        S3_zero_st = 1;
        S3_zero();
        }
      } else {

      }
    } else {

    }
    if ((int )D_print_st == 0) {
      {
	tmp___3 = __VERIFIER_nondet_int();
      }
      if (tmp___3) {
        {
        D_print_st = 1;
        D_print();
        }
      } else {

      }
    } else {

    }
  }
  while_0_break: /* CIL Label */ ;
  }

  return;
}
}
void start_simulation(void) 
{ int kernel_st ;

  {
  kernel_st = 0;
  if ((int )main_in1_req_up == 1) {
    if (main_in1_val != main_in1_val_t) {
      main_in1_val = main_in1_val_t;
      main_in1_ev = 0;
    } else {

    }
    main_in1_req_up = 0;
  } else {

  }
  if ((int )main_in2_req_up == 1) {
    if (main_in2_val != main_in2_val_t) {
      main_in2_val = main_in2_val_t;
      main_in2_ev = 0;
    } else {

    }
    main_in2_req_up = 0;
  } else {

  }
  if ((int )main_sum_req_up == 1) {
    if (main_sum_val != main_sum_val_t) {
      main_sum_val = main_sum_val_t;
      main_sum_ev = 0;
    } else {

    }
    main_sum_req_up = 0;
  } else {

  }
  if ((int )main_diff_req_up == 1) {
    if (main_diff_val != main_diff_val_t) {
      main_diff_val = main_diff_val_t;
      main_diff_ev = 0;
    } else {

    }
    main_diff_req_up = 0;
  } else {

  }
  if ((int )main_pres_req_up == 1) {
    if (main_pres_val != main_pres_val_t) {
      main_pres_val = main_pres_val_t;
      main_pres_ev = 0;
    } else {

    }
    main_pres_req_up = 0;
  } else {

  }
  if ((int )main_dbl_req_up == 1) {
    if (main_dbl_val != main_dbl_val_t) {
      main_dbl_val = main_dbl_val_t;
      main_dbl_ev = 0;
    } else {

    }
    main_dbl_req_up = 0;
  } else {

  }
  if ((int )main_zero_req_up == 1) {
    if (main_zero_val != main_zero_val_t) {
      main_zero_val = main_zero_val_t;
      main_zero_ev = 0;
    } else {

    }
    main_zero_req_up = 0;
  } else {

  }
  if ((int )main_clk_req_up == 1) {
    if ((int )main_clk_val != (int )main_clk_val_t) {
      main_clk_val = main_clk_val_t;
      main_clk_ev = 0;
      if ((int )main_clk_val == 1) {
        main_clk_pos_edge = 0;
        main_clk_neg_edge = 2;
      } else {
        main_clk_neg_edge = 0;
        main_clk_pos_edge = 2;
      }
    } else {

    }
    main_clk_req_up = 0;
  } else {

  }
  if ((int )N_generate_i == 1) {
    N_generate_st = 0;
  } else {
    N_generate_st = 2;
  }
  if ((int )S1_addsub_i == 1) {
    S1_addsub_st = 0;
  } else {
    S1_addsub_st = 2;
  }
  if ((int )S2_presdbl_i == 1) {
    S2_presdbl_st = 0;
  } else {
    S2_presdbl_st = 2;
  }
  if ((int )S3_zero_i == 1) {
    S3_zero_st = 0;
  } else {
    S3_zero_st = 2;
  }
  if ((int )D_print_i == 1) {
    D_print_st = 0;
  } else {
    D_print_st = 2;
  }
  if ((int )main_in1_ev == 0) {
    main_in1_ev = 1;
  } else {

  }
  if ((int )main_in2_ev == 0) {
    main_in2_ev = 1;
  } else {

  }
  if ((int )main_sum_ev == 0) {
    main_sum_ev = 1;
  } else {

  }
  if ((int )main_diff_ev == 0) {
    main_diff_ev = 1;
  } else {

  }
  if ((int )main_pres_ev == 0) {
    main_pres_ev = 1;
  } else {

  }
  if ((int )main_dbl_ev == 0) {
    main_dbl_ev = 1;
  } else {

  }
  if ((int )main_zero_ev == 0) {
    main_zero_ev = 1;
  } else {

  }
  if ((int )main_clk_ev == 0) {
    main_clk_ev = 1;
  } else {

  }
  if ((int )main_clk_pos_edge == 0) {
    main_clk_pos_edge = 1;
  } else {

  }
  if ((int )main_clk_neg_edge == 0) {
    main_clk_neg_edge = 1;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    N_generate_st = 0;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    S1_addsub_st = 0;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    S2_presdbl_st = 0;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    S3_zero_st = 0;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    D_print_st = 0;
  } else {

  }
  if ((int )main_in1_ev == 1) {
    main_in1_ev = 2;
  } else {

  }
  if ((int )main_in2_ev == 1) {
    main_in2_ev = 2;
  } else {

  }
  if ((int )main_sum_ev == 1) {
    main_sum_ev = 2;
  } else {

  }
  if ((int )main_diff_ev == 1) {
    main_diff_ev = 2;
  } else {

  }
  if ((int )main_pres_ev == 1) {
    main_pres_ev = 2;
  } else {

  }
  if ((int )main_dbl_ev == 1) {
    main_dbl_ev = 2;
  } else {

  }
  if ((int )main_zero_ev == 1) {
    main_zero_ev = 2;
  } else {

  }
  if ((int )main_clk_ev == 1) {
    main_clk_ev = 2;
  } else {

  }
  if ((int )main_clk_pos_edge == 1) {
    main_clk_pos_edge = 2;
  } else {

  }
  if ((int )main_clk_neg_edge == 1) {
    main_clk_neg_edge = 2;
  } else {

  }
  {
  while (1) {
    while_1_continue: /* CIL Label */ ;
    {
    kernel_st = 1;
    eval();
    }
    kernel_st = 2;
    if ((int )main_in1_req_up == 1) {
      if (main_in1_val != main_in1_val_t) {
        main_in1_val = main_in1_val_t;
        main_in1_ev = 0;
      } else {

      }
      main_in1_req_up = 0;
    } else {

    }
    if ((int )main_in2_req_up == 1) {
      if (main_in2_val != main_in2_val_t) {
        main_in2_val = main_in2_val_t;
        main_in2_ev = 0;
      } else {

      }
      main_in2_req_up = 0;
    } else {

    }
    if ((int )main_sum_req_up == 1) {
      if (main_sum_val != main_sum_val_t) {
        main_sum_val = main_sum_val_t;
        main_sum_ev = 0;
      } else {

      }
      main_sum_req_up = 0;
    } else {

    }
    if ((int )main_diff_req_up == 1) {
      if (main_diff_val != main_diff_val_t) {
        main_diff_val = main_diff_val_t;
        main_diff_ev = 0;
      } else {

      }
      main_diff_req_up = 0;
    } else {

    }
    if ((int )main_pres_req_up == 1) {
      if (main_pres_val != main_pres_val_t) {
        main_pres_val = main_pres_val_t;
        main_pres_ev = 0;
      } else {

      }
      main_pres_req_up = 0;
    } else {

    }
    if ((int )main_dbl_req_up == 1) {
      if (main_dbl_val != main_dbl_val_t) {
        main_dbl_val = main_dbl_val_t;
        main_dbl_ev = 0;
      } else {

      }
      main_dbl_req_up = 0;
    } else {

    }
    if ((int )main_zero_req_up == 1) {
      if (main_zero_val != main_zero_val_t) {
        main_zero_val = main_zero_val_t;
        main_zero_ev = 0;
      } else {

      }
      main_zero_req_up = 0;
    } else {

    }
    if ((int )main_clk_req_up == 1) {
      if ((int )main_clk_val != (int )main_clk_val_t) {
        main_clk_val = main_clk_val_t;
        main_clk_ev = 0;
        if ((int )main_clk_val == 1) {
          main_clk_pos_edge = 0;
          main_clk_neg_edge = 2;
        } else {
          main_clk_neg_edge = 0;
          main_clk_pos_edge = 2;
        }
      } else {

      }
      main_clk_req_up = 0;
    } else {

    }
    kernel_st = 3;
    if ((int )main_in1_ev == 0) {
      main_in1_ev = 1;
    } else {

    }
    if ((int )main_in2_ev == 0) {
      main_in2_ev = 1;
    } else {

    }
    if ((int )main_sum_ev == 0) {
      main_sum_ev = 1;
    } else {

    }
    if ((int )main_diff_ev == 0) {
      main_diff_ev = 1;
    } else {

    }
    if ((int )main_pres_ev == 0) {
      main_pres_ev = 1;
    } else {

    }
    if ((int )main_dbl_ev == 0) {
      main_dbl_ev = 1;
    } else {

    }
    if ((int )main_zero_ev == 0) {
      main_zero_ev = 1;
    } else {

    }
    if ((int )main_clk_ev == 0) {
      main_clk_ev = 1;
    } else {

    }
    if ((int )main_clk_pos_edge == 0) {
      main_clk_pos_edge = 1;
    } else {

    }
    if ((int )main_clk_neg_edge == 0) {
      main_clk_neg_edge = 1;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      N_generate_st = 0;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      S1_addsub_st = 0;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      S2_presdbl_st = 0;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      S3_zero_st = 0;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      D_print_st = 0;
    } else {

    }
    if ((int )main_in1_ev == 1) {
      main_in1_ev = 2;
    } else {

    }
    if ((int )main_in2_ev == 1) {
      main_in2_ev = 2;
    } else {

    }
    if ((int )main_sum_ev == 1) {
      main_sum_ev = 2;
    } else {

    }
    if ((int )main_diff_ev == 1) {
      main_diff_ev = 2;
    } else {

    }
    if ((int )main_pres_ev == 1) {
      main_pres_ev = 2;
    } else {

    }
    if ((int )main_dbl_ev == 1) {
      main_dbl_ev = 2;
    } else {

    }
    if ((int )main_zero_ev == 1) {
      main_zero_ev = 2;
    } else {

    }
    if ((int )main_clk_ev == 1) {
      main_clk_ev = 2;
    } else {

    }
    if ((int )main_clk_pos_edge == 1) {
      main_clk_pos_edge = 2;
    } else {

    }
    if ((int )main_clk_neg_edge == 1) {
      main_clk_neg_edge = 2;
    } else {

    }
    if ((int )N_generate_st == 0) {

    } else {
      if ((int )S1_addsub_st == 0) {

      } else {
        if ((int )S2_presdbl_st == 0) {

        } else {
          if ((int )S3_zero_st == 0) {

          } else {
            if ((int )D_print_st == 0) {

            } else {
              goto while_1_break;
            }
          }
        }
      }
    }
  }
  while_1_break: /* CIL Label */ ;
  }

  return;
}
}
int main(void) 
{ int count ;
  int __retres2 ;

  {
  {

  main_in1_ev  =    2;
  main_in1_req_up  =    0;
  main_in2_ev  =    2;
  main_in2_req_up  =    0;
  main_diff_ev  =    2;
  main_diff_req_up  =    0;
  main_sum_ev  =    2;
  main_sum_req_up  =    0;
  main_pres_ev  =    2;
  main_pres_req_up  =    0;
  main_dbl_ev  =    2;
  main_dbl_req_up  =    0;
  main_zero_ev  =    2;
  main_zero_req_up  =    0;
  main_clk_val  =    0;
  main_clk_ev  =    2;
  main_clk_req_up  =    0;
  main_clk_pos_edge  =    2;
  main_clk_neg_edge  =    2;


  count = 0;
  N_generate_i = 0;
  S1_addsub_i = 0;
  S2_presdbl_i = 0;
  S3_zero_i = 0;
  D_print_i = 0;
  start_simulation();
  }
  {
  while (1) {
    while_2_continue: /* CIL Label */ ;
    {
    main_clk_val_t = 1;
    main_clk_req_up = 1;
    start_simulation();
    count += 1;
    }
    if (count == 5) {
      if (! (D_z == 0)) {
        {
        error();
        }
      } else {

      }
      count = 0;
    } else {

    }
    {
    main_clk_val_t = 0;
    main_clk_req_up = 1;
    start_simulation();
    }
  }
  while_2_break: /* CIL Label */ ;
  }
  __retres2 = 0;
  return (__retres2);
}
}
