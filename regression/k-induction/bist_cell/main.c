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

int b0_val  ;
int b0_val_t  ;
int b0_ev  ;
int b0_req_up  ;
int b1_val  ;
int b1_val_t  ;
int b1_ev  ;
int b1_req_up  ;
int d0_val  ;
int d0_val_t  ;
int d0_ev  ;
int d0_req_up  ;
int d1_val  ;
int d1_val_t  ;
int d1_ev  ;
int d1_req_up  ;
int z_val  ;
int z_val_t  ;
int z_ev  ;
int z_req_up  ;
int comp_m1_st  ;
int comp_m1_i  ;
void method1(void) 
{ int s1 ;
  int s2 ;
  int s3 ;

  {
  if (b0_val) {
    if (d1_val) {
      s1 = 0;
    } else {
      s1 = 1;
    }
  } else {
    s1 = 1;
  }
  if (d0_val) {
    if (b1_val) {
      s2 = 0;
    } else {
      s2 = 1;
    }
  } else {
    s2 = 1;
  }
  if (s2) {
    s3 = 0;
  } else {
    if (s1) {
      s3 = 0;
    } else {
      s3 = 1;
    }
  }
  if (s2) {
    if (s1) {
      s2 = 1;
    } else {
      s2 = 0;
    }
  } else {
    s2 = 0;
  }
  if (s2) {
    z_val_t = 0;
  } else {
    if (s3) {
      z_val_t = 0;
    } else {
      z_val_t = 1;
    }
  }
  z_req_up = 1;
  comp_m1_st = 2;

  return;
}
}
int is_method1_triggered(void) 
{ int __retres1 ;

  {
  if ((int )b0_ev == 1) {
    __retres1 = 1;
    goto return_label;
  } else {
    if ((int )b1_ev == 1) {
      __retres1 = 1;
      goto return_label;
    } else {
      if ((int )d0_ev == 1) {
        __retres1 = 1;
        goto return_label;
      } else {
        if ((int )d1_ev == 1) {
          __retres1 = 1;
          goto return_label;
        } else {

        }
      }
    }
  }
  __retres1 = 0;
  return_label: /* CIL Label */ 
  return (__retres1);
}
}
void update_b0(void) 
{ 

  {
  if ((int )b0_val != (int )b0_val_t) {
    b0_val = b0_val_t;
    b0_ev = 0;
  } else {

  }
  b0_req_up = 0;

  return;
}
}
void update_b1(void) 
{ 

  {
  if ((int )b1_val != (int )b1_val_t) {
    b1_val = b1_val_t;
    b1_ev = 0;
  } else {

  }
  b1_req_up = 0;

  return;
}
}
void update_d0(void) 
{ 

  {
  if ((int )d0_val != (int )d0_val_t) {
    d0_val = d0_val_t;
    d0_ev = 0;
  } else {

  }
  d0_req_up = 0;

  return;
}
}
void update_d1(void) 
{ 

  {
  if ((int )d1_val != (int )d1_val_t) {
    d1_val = d1_val_t;
    d1_ev = 0;
  } else {

  }
  d1_req_up = 0;

  return;
}
}
void update_z(void) 
{ 

  {
  if ((int )z_val != (int )z_val_t) {
    z_val = z_val_t;
    z_ev = 0;
  } else {

  }
  z_req_up = 0;

  return;
}
}
void update_channels(void) 
{ 

  {
  if ((int )b0_req_up == 1) {
    {
    update_b0();
    }
  } else {

  }
  if ((int )b1_req_up == 1) {
    {
    update_b1();
    }
  } else {

  }
  if ((int )d0_req_up == 1) {
    {
    update_d0();
    }
  } else {

  }
  if ((int )d1_req_up == 1) {
    {
    update_d1();
    }
  } else {

  }
  if ((int )z_req_up == 1) {
    {
    update_z();
    }
  } else {

  }

  return;
}
}
void init_threads(void) 
{ 

  {
  if ((int )comp_m1_i == 1) {
    comp_m1_st = 0;
  } else {
    comp_m1_st = 2;
  }

  return;
}
}
int exists_runnable_thread(void) 
{ int __retres1 ;

  {
  if ((int )comp_m1_st == 0) {
    __retres1 = 1;
    goto return_label;
  } else {

  }
  __retres1 = 0;
  return_label: /* CIL Label */ 
  return (__retres1);
}
}
void eval(void) 
{ int tmp ;
  int tmp___0 ;
 // int __VERIFIER_nondet_int(); 

  {
  {
  while (1) {
    while_0_continue: /* CIL Label */ ;
    {
    tmp___0 = exists_runnable_thread();
    }
    if (tmp___0) {

    } else {
      goto while_0_break;
    }
    if ((int )comp_m1_st == 0) {
      {
	tmp = __VERIFIER_nondet_int(); 
      }
      if (tmp) {
        {
        comp_m1_st = 1;
        method1();
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
void fire_delta_events(void) 
{ 

  {
  if ((int )b0_ev == 0) {
    b0_ev = 1;
  } else {

  }
  if ((int )b1_ev == 0) {
    b1_ev = 1;
  } else {

  }
  if ((int )d0_ev == 0) {
    d0_ev = 1;
  } else {

  }
  if ((int )d1_ev == 0) {
    d1_ev = 1;
  } else {

  }
  if ((int )z_ev == 0) {
    z_ev = 1;
  } else {

  }

  return;
}
}
void reset_delta_events(void) 
{ 

  {
  if ((int )b0_ev == 1) {
    b0_ev = 2;
  } else {

  }
  if ((int )b1_ev == 1) {
    b1_ev = 2;
  } else {

  }
  if ((int )d0_ev == 1) {
    d0_ev = 2;
  } else {

  }
  if ((int )d1_ev == 1) {
    d1_ev = 2;
  } else {

  }
  if ((int )z_ev == 1) {
    z_ev = 2;
  } else {

  }

  return;
}
}
void activate_threads(void) 
{ int tmp ;

  {
  {
  tmp = is_method1_triggered();
  }
  if (tmp) {
    comp_m1_st = 0;
  } else {

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
    while_1_continue: /* CIL Label */ ;
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
      goto while_1_break;
    } else {

    }
  }
  while_1_break: /* CIL Label */ ;
  }

  return;
}
}
void init_model(void) 
{ 

  {
  b0_val = 0;
  b0_ev = 2;
  b0_req_up = 0;
  b1_val = 0;
  b1_ev = 2;
  b1_req_up = 0;
  d0_val = 0;
  d0_ev = 2;
  d0_req_up = 0;
  d1_val = 0;
  d1_ev = 2;
  d1_req_up = 0;
  z_val = 0;
  z_ev = 2;
  z_req_up = 0;
  b0_val_t = 1;
  b0_req_up = 1;
  b1_val_t = 1;
  b1_req_up = 1;
  d0_val_t = 1;
  d0_req_up = 1;
  d1_val_t = 1;
  d1_req_up = 1;
  comp_m1_i = 0;

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
  if (! ((int )z_val == 0)) {
    {
    error();
    }
  } else {

  }
  __retres1 = 0;
  return (__retres1);
}
}
