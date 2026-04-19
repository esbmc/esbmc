#include <pthread.h>
#include <global_vars.h>
#include <assert.h>

#include <ecsc.h>

/*---- Types ----*/
/* Structure forward decls */

/*---- Global variables ----*/

/*---- Events from Process Sensitivity Lists ----*/
_Bool wait_example_event_1; /*0x9caea38*/
_Bool wait_example_event_2; /*0x9c89c10*/
_Bool wait_example_event_3; /*0x9cfa1b0*/

struct {
	_Bool e1;
	_Bool e2;
} events;

/*---- Properties ----*/

void check_property()
{
  if (events.e1 && events.e2)
    assert(0);
}

/*---- Threads ----*/

void *wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev(void *arg)
{

	int this_addr;
    int alloca_20_point;
    int tmp__1;
    int tmp__2;
    int tmp__3;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev *my_struct;
    my_struct = (struct inputs_wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev *) arg;
    alloca_20_point = 0;
    this_addr = this;
    goto bb;

bb:
    tmp__1 = this_addr;
    tmp__2 = this_addr;
    wait_event(wait_example_event_1);
    tmp__3 = this_addr;
    wait_example_event_3 = notify_event();
	events.e2=1;
	check_property();
    goto bb;

    wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev_join = 1;

    pthread_exit(NULL);
}


void *wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev(void *arg)
{

	int this_addr;
    int alloca_20_point;
    int tmp__4;
    int tmp__5;
    int tmp__6;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev *my_struct;
    my_struct = (struct inputs_wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev *) arg;
    alloca_20_point = 0;
    this_addr = this;
    goto bb;

bb:
    tmp__4 = this_addr;
    wait_example_event_1 = notify_event();
    tmp__5 = this_addr;
    tmp__6 = this_addr;
    wait_event(wait_example_event_3); 
	events.e1=1;
	check_property();
    goto bb;
  
	wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev_join = 1;

    pthread_exit(NULL);
}


_Bool nondet_bool();  // returns non-deterministic boolean value

/**********************main C program***********************/
int main(){


    /* Both processes are SC_THREAD types so must be invoked by default at the start 
     * However if either is halted (such as by a wait command) then it must be resumed whenever the wait expires (such as when an event is notified)	
*/	
    pthread_t id0;
    pthread_t id1;
    pthread_t id2;

    pthread_create(&id1, NULL, wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev,  &my_inputs_wait_example__ZN12wait_example8do_test1Ev_pid1_ZN12wait_example8do_test1Ev);
  
    pthread_create(&id0, NULL, wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev, &my_inputs_wait_example__ZN12wait_example8do_test2Ev_pid0_ZN12wait_example8do_test2Ev);

}
