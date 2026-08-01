#if 0
/* Precomputed transition data */
States:
label	id	final
0	init	1
1	T0_2	0

Symbol table:
id	symbol			cexpr
0	_ltl2ba_cexpr_0_status	{ pressed }
1	_ltl2ba_cexpr_1_status	{ charge > min }

Stuttering:


!{pressed}&!{charge > min}
Transitions:
1	1	
0	1	

Reachability:
1	1	
0	1	

Accepting cycles: {0}
Accepting states: {0}

{pressed}&!{charge > min}
Transitions:
0	1	
0	1	

Reachability:
0	1	
0	1	

Accepting cycles: {}
Accepting states: {}

!{pressed}&{charge > min}
Transitions:
1	1	
1	1	

Reachability:
1	1	
1	1	

Accepting cycles: {0}
Accepting states: {0,1}

{pressed}&{charge > min}
Transitions:
1	1	
1	1	

Reachability:
1	1	
1	1	

Accepting cycles: {0}
Accepting states: {0,1}


Optimistic transitions:
1	1	
1	1	
Optimistic reachability:
1	1	
1	1	

Accepting optimistic cycles: {0}
Accepting optimistic states: {0,1}


Pessimistic transitions:
 0: {1}
 1: {1}


Pessimistic reachable:
 0: {1}
 1: {1}

Accepting pessimistic cycles: {}
Accepting pessimistic states: {}
#endif
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

void __ESBMC_switch_to_monitor(void);
void __ESBMC_switch_from_monitor(void);
void __ESBMC_register_monitor(pthread_t t);
void __ESBMC_really_atomic_begin();
void __ESBMC_really_atomic_end();
void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();
void __ESBMC_assume(_Bool prop);
void __ESBMC_kill_monitor();
unsigned int nondet_uint();

extern int pressed;
extern int charge, min;

char __ESBMC_property__ltl2ba_cexpr_0[] = "pressed";
int _ltl2ba_cexpr_0_status(void) { return pressed; }
char __ESBMC_property__ltl2ba_cexpr_1[] = "charge > min";
int _ltl2ba_cexpr_1_status(void) { return charge > min; }

typedef enum {
	_ltl2ba_state_0,
	_ltl2ba_state_1,
} _ltl2ba_state;

_ltl2ba_state _ltl2ba_statevar =_ltl2ba_state_0;

unsigned int _ltl2ba_visited_states[2];

void
ltl2ba_fsm(bool state_stats, unsigned int num_iters)
{
	unsigned int choice;
	unsigned int iters;
	_Bool state_is_viable;

	/* Original formula:
	 * G({pressed} -> F {charge > min})
	 */

	for (iters = 0; iters < num_iters; iters++) {
		choice = nondet_uint();

		__ESBMC_atomic_begin();

		switch(_ltl2ba_statevar) {
		case _ltl2ba_state_0:
			state_is_viable = (((!(pressed)) || ((charge > min))) || ((1)) || (((charge > min))) || ((false)));
			if (choice == 0) {
				__ESBMC_assume(((!(pressed)) || ((charge > min))));
				_ltl2ba_statevar = _ltl2ba_state_0;
			} else if (choice == 1) {
				__ESBMC_assume(((1)));
				_ltl2ba_statevar = _ltl2ba_state_1;
			} else if (choice == 2) {
				__ESBMC_assume((((charge > min))));
				_ltl2ba_statevar = _ltl2ba_state_0;
			} else {
				__ESBMC_assume(0);
			}
			break;
		case _ltl2ba_state_1:
			state_is_viable = (((1)) || (((charge > min))) || ((false)));
			if (choice == 0) {
				__ESBMC_assume(((1)));
				_ltl2ba_statevar = _ltl2ba_state_1;
			} else if (choice == 1) {
				__ESBMC_assume((((charge > min))));
				_ltl2ba_statevar = _ltl2ba_state_0;
			} else {
				__ESBMC_assume(0);
			}
			break;
		}
		if (state_stats)
			_ltl2ba_visited_states[_ltl2ba_statevar]++;

		__ESBMC_atomic_end();
		// __ESBMC_switch_from_monitor();
	}

	__ESBMC_assert(num_iters == iters, "Unwind bound on ltl2ba_fsm insufficient");

	return;
}

#ifndef LTL_PREFIX_BOUND
#define LTL_PREFIX_BOUND 2147483648
#endif

#define max(x,y) ((x) < (y) ? (y) : (x))

void * ltl2ba_thread(void *dummy)
{
	ltl2ba_fsm(false, LTL_PREFIX_BOUND);
	return 0;
	(void)dummy;
}

void ltl2ba_start_monitor(void)
{
	pthread_t t;

//	__ESBMC_atomic_begin();
	pthread_create(&t, NULL, ltl2ba_thread, NULL);
	__ESBMC_register_monitor(t);
//	__ESBMC_atomic_end();

	// __ESBMC_switch_to_monitor();
}

_Bool _ltl2ba_stutter_accept_table[4][2] = {
{
  true, false, 
},
{
  false, false, 
},
{
  true, true, 
},
{
  true, true, 
},
};

_Bool _ltl2ba_good_prefix_excluded_states[2] = {
true, true, 
};

_Bool _ltl2ba_bad_prefix_states[2] = {
false, false, 
};

unsigned int
_ltl2ba_sym_to_idx(void)
{
	unsigned int idx = 0;
	idx |= ((pressed)) ? 1 : 0;
	idx |= ((charge > min)) ? 2 : 0;
	return idx;
}

void
ltl2ba_finish_monitor(void)
{
	__ESBMC_kill_monitor();

	__ESBMC_assert(!_ltl2ba_bad_prefix_states[_ltl2ba_statevar],"LTL_BAD");

	__ESBMC_assert(!_ltl2ba_stutter_accept_table[_ltl2ba_sym_to_idx()][_ltl2ba_statevar],"LTL_FAILING");

	__ESBMC_assert(!_ltl2ba_good_prefix_excluded_states[_ltl2ba_statevar],"LTL_SUCCEEDING");

	return;
}
