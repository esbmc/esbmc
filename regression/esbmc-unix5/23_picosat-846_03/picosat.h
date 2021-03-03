/****************************************************************************
Copyright (c) 2006 - 2008, Armin Biere, Johannes Kepler University.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
****************************************************************************/

#ifndef picosat_h_INCLUDED
#define picosat_h_INCLUDED

/*------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>

/*------------------------------------------------------------------------*/
/* These are the return values for 'picosat_sat' as for instance
 * standardized by the output format of the SAT competition.
 */
#define PICOSAT_UNKNOWN		0
#define PICOSAT_SATISFIABLE	10
#define PICOSAT_UNSATISFIABLE	20

/*------------------------------------------------------------------------*/

const char *picosat_version (void);
const char *picosat_config (void);
const char *picosat_copyright (void);

/*------------------------------------------------------------------------*/
/* You can make picosat use an external memory manager instead of the one
 * provided by LIBC. But then you need to call these three function before
 * 'picosat_init'.  The memory manager functions here all have an additional
 * first argument which is a pointer to the memory manager, but otherwise
 * are supposed to work as their LIBC counter parts 'malloc', 'realloc' and
 * 'free'.  As exception the 'resize' and 'delete' function have as third
 * argument the number of bytes of the block given as second argument.
 */
void picosat_set_new (void * mgr, void * (*)(void *, size_t));
void picosat_set_resize (void *, void * (*)(void *, void *, size_t, size_t));
void picosat_set_delete (void *, void (*)(void *, void *, size_t));

/*------------------------------------------------------------------------*/

void picosat_init (void);		/* constructor */
void picosat_reset (void);		/* destructor */

/*------------------------------------------------------------------------*/
/* The following five functions are essentially parameters to 'init', and
 * thus should be called right after 'picosat_init' before doing anything
 * else.  You should not call any of them after adding a literal.
 */

/* Set output file, default is 'stdout'.
 */
void picosat_set_output (FILE *);

/* Measure all time spent in all calls in the solver.  By default only the
 * time spent in 'picosat_sat' is measured.  Enabling this function may for
 * instance tripple the time needed to add large CNFs, since every call to
 * 'picosat_add' will trigger a call to 'getrusage'.
 */
void picosat_measure_all_calls (void);

/* Set the prefix used for printing verbose messages and statistics.
 * Default is "c ".
 */
void picosat_set_prefix (const char *);

/* The function 'picosat_set_incremental_rup_file' produces
 * a proof trace in RUP format on the fly.  The resulting RUP file may
 * contain learned clauses that are not actual in the clausal core.
 */

/* Set verbosity level.  A verbosity level of 1 and above prints more and
 * more detailed progress reports on the output file, set by
 * 'picosat_set_output'.  Verbose messages are prefixed with the string set
 * by 'picosat_set_prefix'.
 */
void picosat_set_verbosity (int new_verbosity_level);

/* Set default initial phase: 
 *
 *   negative = false
 *
 *   posivie  = true
 *
 *   0        = Jeroslow-Wang (default)
 *
 * After a variable has been assigned the first time, it will always
 * be assigned the previous value if it is picked as decision variable.
 * The initial assignment can be choosen with this function.
 */
void picosat_set_global_default_phase (int);

/* Set next/initial phase of a particular variable if picked as decision
 * variable.  Second argument 'phase' has the following meaning:
 *
 *   negative = next value if picked as decision variable is false
 *
 *   positive = next value if picked as decision variable is true
 *
 *   0        = use global default phase as next value and
 *              assume 'lit' was never assigned
 *
 * Again if 'lit' is assigned afterwards through a forced assignment,
 * then this forced assignment is the next phase if this variable is
 * used as decision variable.
 */
void picosat_set_default_phase_lit (int lit, int phase);

/* Allows to print to internal 'out' file from client.
 */
void picosat_message (int verbosity_level, const char * fmt, ...);

/* Deprecated!
 */
#define picosat_enable_verbosity() picosat_set_verbosity (1)

/* Set a seed for the random number generator.  The random number generator
 * is currently just used for generating random decisions.  In our
 * experiments having random decisions did not really help on industrial
 * examples, but was rather helpful to randomize the solver in order to
 * do proper benchmarking of different internal parameter sets.
 */
void picosat_set_seed (unsigned random_number_generator_seed);

/* If you ever want to extract cores or proof traces with the current
 * instance of PicoSAT initialized with 'picosat_init', then make sure to
 * call 'picosat_enable_trace_generation' right after 'picosat_init'.   This
 * is not necessary if you only use 'picosat_set_incremental_rup_file'.
 *
 * NOTE, trace generation code is not necessarily included, e.g. if you
 * configure picosat with full optimzation as './configure -O' or with
 * './configure --no-trace'.  This speeds up the solver slightly.  Then you
 * you do not get any results by trying to generate traces.
 */
void picosat_enable_trace_generation (void);

/* You can dump proof traces in RUP format incrementally even without
 * keeping the proof trace in memory.  The advantage is a reduction of
 * memory usage, but the dumped clauses do not necessarily belong to the
 * clausal core.  Beside the file the additional parameters denotes the
 * maximal number of variables and the number of original clauses.
 */
void picosat_set_incremental_rup_file (FILE * file, int m, int n);

/*------------------------------------------------------------------------*/
/* This function returns the next available unused variable index and
 * allocates a variable for it even though this variable does not occur as
 * assumption, nor in a clause or any other constraints.  In future calls to
 * 'picosat_sat', 'picosat_deref' and particularly for 'picosat_changed',
 * this variable is treated as if it had been used.
 */
int picosat_inc_max_var (void);

/*------------------------------------------------------------------------*/
/* If you know a good estimate on how many variables you are going to use
 * then calling this function before adding literals will result in less
 * resizing of the variable table.  But this is just a minor optimization.
 * Beside exactly allocating enough variables it has the same effect as
 * calling 'picosat_inc_max_var'.
 */
void picosat_adjust (int max_idx);

/*------------------------------------------------------------------------*/
/* Statistics.
 */
int picosat_variables (void);				/* p cnf <m> n */
int picosat_added_original_clauses (void);		/* p cnf m <n> */
size_t picosat_max_bytes_allocated (void);
double picosat_time_stamp (void);			/* ... in process */
void picosat_stats (void);				/* > output file */

/* The time spent in the library or in 'picosat_sat'.  The former is only
 * returned if, right after initialization 'picosat_measure_all_calls'
 * is called.
 */
double picosat_seconds (void);

/*------------------------------------------------------------------------*/
/* Add a literal of the next clause.  A zero terminates the clause.  The
 * solver is incremental.  Adding a new literal will reset the previous
 * assignment.
 */
void picosat_add (int lit);

/* Print the CNF to the given file in DIMACS format.
 */
void picosat_print (FILE *);

/* You can add arbitrary many assertions before the next 'picosat_sat'.
 * An assumption is only valid for the next 'picosat_sat' and will be taken
 * back afterwards.  Adding a new assumption will reset the previous
 * assignment.
 */
void picosat_assume (int lit);

/*------------------------------------------------------------------------*/
/* This is an experimental feature for handling 'all different constraints'
 * (ADC).  Currently only one global ADC can be handled.  The bit-width of
 * all the bit-vectors entered in this ADC (stored in 'all different
 * objects' or ADOs) has to be identical.
 *
 * TODO: also handle top level assigned literals here.
 */
void picosat_add_ado_lit (int);

/*------------------------------------------------------------------------*/
/* Call the main SAT routine.  A negative decision limits sets no limit on
 * the number of decisions.  The return values are as above, e.g.
 * 'PICOSAT_UNSATISFIABLE', 'PICOSAT_SATISFIABLE', or 'PICOSAT_UNKNOWN'.
 */
int picosat_sat (int decision_limit);

/* After 'picosat_sat' was called and returned 'PICOSAT_SATISFIABLE', then
 * the satisfying assignment can be obtained by 'dereferencing' literals.
 * The value of the literal is return as '1' for 'true',  '-1' for 'false'
 * and '0' for an unknown value.
 */
int picosat_deref (int lit);

/* Same as before but just returns true resp. false if the literals is
 * forced to this assignment at the top level.  This function does not
 * require that 'picosat_sat' was called and also does not internally reset
 * incremental usage.
 */
int picosat_deref_toplevel (int lit);

/* Returns non zero if the CNF is unsatisfiable because an empty clause was
 * added or derived.
 */
int picosat_inconsistent  (void);

/*------------------------------------------------------------------------*/
/* Assume that a previous call to 'picosat_sat' in incremental usage,
 * returned 'SATISFIABLE'.  Then a couple of clauses and optionally new
 * variables were added (a new variable is a variable that has an index
 * larger then the maximum variable added so far).  The next call to
 * 'picosat_sat' also returns 'SATISFIABLE'. If this function
 * 'picosat_changed' returns '0', then the assignment to the old variables
 * did not change.  Otherwise it may have changed.   The return value to
 * this function is only valid until new clauses are added through
 * 'picosat_add', an assumption is made through 'picosat_assume', or again
 * 'picosat_sat' is called.  This is the same assumption as for
 * 'picosat_deref'.
 *
 * TODO currently this function may also return a non zero value even if the
 * old assignment did not change, because it only checks whether the
 * assignment of at least one old variable was flipped at least once during
 * the search.  In principle it should be possible to be exact in the other
 * direcetion as well by using a counter of variables that have an odd
 * number of flips.  But this is not implemented yet.
 */
int picosat_changed (void);

/*------------------------------------------------------------------------*/
/* The following five functions internally extract the variable and clausal
 * core and thus require trace generation to be enabled with
 * 'picosat_enable_trace_generation' right after calling 'picosat_init'.
 *
 * TODO: using these functions in incremental mode with failed assumptions
 * has only been tested for 'picosat_corelit' thoroughly.  The others may
 * only work in non-incremental mode or without using 'picosat_assume'.
 */

/* This function gives access to the variable core, which is made up of the
 * variables that were resolved in deriving the empty clauses.
 */
int picosat_corelit (int lit);

/* Write the clauses that were used in deriving the empty clause to a file
 * in DIMACS format.
 */
void picosat_write_clausal_core (FILE * core_file);

/* Write a proof trace in TraceCheck format to a file.
 */
void picosat_write_compact_trace (FILE * trace_file);
void picosat_write_extended_trace (FILE * trace_file);

/* Write a RUP trace to a file.  This trace file contains only the learned
 * core clauses while this is not necessarily the case for the RUP file
 * obtained with 'picosat_set_incremental_rup_file'.
 */
void picosat_write_rup_trace (FILE * trace_file);

/*------------------------------------------------------------------------*/
/* Keeping the proof trace around is not necessary if an over-approximation
 * of the core is enough.  A literal is 'used' if it was involved in a
 * resolution to derive a learned clause.  The core literals are necessarily
 * a subset of the 'used' literals.
 */

int picosat_usedlit (int lit);
/*------------------------------------------------------------------------*/
#endif
