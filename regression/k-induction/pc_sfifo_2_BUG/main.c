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

int q_buf_0  ;
int q_free  ;
int q_read_ev  ;
int q_write_ev  ;
int q_req_up  ;
int q_ev  ;
void update_fifo_q(void) 
{ 

  {
  if ((int )q_free == 0) {
    q_write_ev = 0;
  } else {

  }
  if ((int )q_free == 1) {
    q_read_ev = 0;
  } else {

  }
  q_ev = 0;
  q_req_up = 0;

  return;
}
}
int p_num_write  ;
int p_last_write  ;
int p_dw_st  ;
int p_dw_pc  ;
int p_dw_i  ;
int c_num_read  ;
int c_last_read  ;
int c_dr_st  ;
int c_dr_pc  ;
int c_dr_i  ;
int is_do_write_p_triggered(void) 
{ int __retres1 ;

  {
  if ((int )p_dw_pc == 1) {
    if ((int )q_read_ev == 1) {
      __retres1 = 1;
      goto return_label;
    } else {

    }
  } else {

  }
  __retres1 = 0;
  return_label: /* CIL Label */ 
  return (__retres1);
}
}
int is_do_read_c_triggered(void) 
{ int __retres1 ;

  {
  if ((int )c_dr_pc == 1) {
    if ((int )q_write_ev == 1) {
      __retres1 = 1;
      goto return_label;
    } else {

    }
  } else {

  }
  __retres1 = 0;
  return_label: /* CIL Label */ 
  return (__retres1);
}
}
void immediate_notify_threads(void) 
{ int tmp ;
  int tmp___0 ;

  {
  {
  tmp = is_do_write_p_triggered();
  }
  if (tmp) {
    p_dw_st = 0;
  } else {

  }
  {
  tmp___0 = is_do_read_c_triggered();
  }
  if (tmp___0) {
    c_dr_st = 0;
  } else {

  }

  return;
}
}
void do_write_p(void) 
{ 
//  int __VERIFIER_nondet_int();

  {
  if ((int )p_dw_pc == 0) {
    goto DW_ENTRY;
  } else {
    if ((int )p_dw_pc == 1) {
      goto DW_WAIT_READ;
    } else {

    }
  }
  DW_ENTRY: 
  {
  while (1) {
    while_0_continue: /* CIL Label */ ;
    if ((int )q_free == 0) {
      p_dw_st = 2;
      p_dw_pc = 1;

      goto return_label;
      DW_WAIT_READ: ;
    } else {

    }
    {
      q_buf_0 = __VERIFIER_nondet_int();
    p_last_write = q_buf_0;
    p_num_write += 1;
    q_free = 0;
    q_req_up = 1;
    }
  }
  while_0_break: /* CIL Label */ ;
  }
  return_label: /* CIL Label */ 
  return;
}
}
static int a_t  ;
void do_read_c(void) 
{ int a ;

  {
  if ((int )c_dr_pc == 0) {
    goto DR_ENTRY;
  } else {
    if ((int )c_dr_pc == 1) {
      goto DR_WAIT_WRITE;
    } else {

    }
  }
  DR_ENTRY: 
  {
  while (1) {
    while_1_continue: /* CIL Label */ ;
    if ((int )q_free == 1) {
      c_dr_st = 2;
      c_dr_pc = 1;
      a_t = a;

      goto return_label;
      DR_WAIT_WRITE: 
      a = a_t;
    } else {

    }
    a = q_buf_0;
    c_last_read = a;
    c_num_read += 1;
    q_free = 1;
    q_req_up = 1;
    if (p_last_write == c_last_read) {
      if (p_num_write == c_num_read) {

      } else {
        {
        error();
        }
      }
    } else {
      {
      error();
      }
    }
  }
  while_1_break: /* CIL Label */ ;
  }
  return_label: /* CIL Label */ 
  return;
}
}
void update_channels(void) 
{ 

  {
  if ((int )q_req_up == 1) {
    {
    update_fifo_q();
    }
  } else {

  }

  return;
}
}
void init_threads(void) 
{ 

  {
  if ((int )p_dw_i == 1) {
    p_dw_st = 0;
  } else {
    p_dw_st = 2;
  }
  if ((int )c_dr_i == 1) {
    c_dr_st = 0;
  } else {
    c_dr_st = 2;
  }

  return;
}
}
int exists_runnable_thread(void) 
{ int __retres1 ;

  {
  if ((int )p_dw_st == 0) {
    __retres1 = 1;
    goto return_label;
  } else {
    if ((int )c_dr_st == 0) {
      __retres1 = 1;
      goto return_label;
    } else {

    }
  }
  __retres1 = 0;
  return_label: /* CIL Label */ 
  return (__retres1);
}
}
void fire_delta_events(void) 
{ 

  {
  if ((int )q_read_ev == 0) {
    q_read_ev = 1;
  } else {

  }
  if ((int )q_write_ev == 0) {
    q_write_ev = 1;
  } else {

  }

  return;
}
}
void reset_delta_events(void) 
{ 

  {
  if ((int )q_read_ev == 1) {
    q_read_ev = 2;
  } else {

  }
  if ((int )q_write_ev == 1) {
    q_write_ev = 2;
  } else {

  }

  return;
}
}
void activate_threads(void) 
{ int tmp ;
  int tmp___0 ;

  {
  {
  tmp = is_do_write_p_triggered();
  }
  if (tmp) {
    p_dw_st = 0;
  } else {

  }
  {
  tmp___0 = is_do_read_c_triggered();
  }
  if (tmp___0) {
    c_dr_st = 0;
  } else {

  }

  return;
}
}
void eval(void) 
{ int tmp ;
  int tmp___0 ;
  int tmp___1 ;
//  int __VERIFIER_nondet_int(); 

  {
  {
  while (1) {
    while_2_continue: /* CIL Label */ ;
    {
    tmp___1 = exists_runnable_thread();
    }
    if (tmp___1) {

    } else {
      goto while_2_break;
    }
    if ((int )p_dw_st == 0) {
      {
	tmp = __VERIFIER_nondet_int();
      }
      if (tmp) {
        {
        p_dw_st = 1;
        do_write_p();
        }
      } else {
	error();
      }
    } else {

    }
    if ((int )c_dr_st == 0) {
      {
	tmp___0 = __VERIFIER_nondet_int();
      }
      if (tmp___0) {
        {
        c_dr_st = 1;
        do_read_c();
        }
      } else {

      }
    } else {

    }
  }
  while_2_break: /* CIL Label */ ;
  }

  return;
}
}
int stop_simulation(void) 
{ int tmp ;
  int __retres2 ;

  {
  {
  tmp = exists_runnable_thread();
  }
  if (tmp) {
    __retres2 = 0;
    goto return_label;
  } else {

  }
  __retres2 = 1;
  return_label: /* CIL Label */ 
  return (__retres2);
}
}
void start_simulation(void) 
{ int kernel_st ;
  int tmp ;

  {
  {
  kernel_st = 0;
  update_channels();
  init_threads();
  fire_delta_events();
  activate_threads();
  reset_delta_events();
  }
  {
  while (1) {
    while_3_continue: /* CIL Label */ ;
    {
    kernel_st = 1;
    eval();
    }
    {
    kernel_st = 2;
    update_channels();
    }
    {
    kernel_st = 3;
    fire_delta_events();
    activate_threads();
    reset_delta_events();
    tmp = stop_simulation();
    }
    if (tmp) {
      goto while_3_break;
    } else {

    }
  }
  while_3_break: /* CIL Label */ ;
  }

  return;
}
}
void init_model(void) 
{ 

  {
  q_free = 1;
  q_write_ev = 2;
  q_read_ev = q_write_ev;
  p_num_write = 0;
  p_dw_pc = 0;
  p_dw_i = 1;
  c_num_read = 0;
  c_dr_pc = 0;
  c_dr_i = 1;

  return;
}
}
int main(void) 
{ int __retres1 ;

  {
  {
  init_model();
  start_simulation();
  }
  __retres1 = 0;
  return (__retres1);
}
}
