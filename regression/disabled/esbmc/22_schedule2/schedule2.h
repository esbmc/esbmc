
/* $Log: schedule.h,v $
 * Revision 1.4  1993/05/04  12:23:44  foster
 * Debug stuff removed
 *
 * Revision 1.3  1993/05/03  20:26:51  foster
 * Full functionality
 *
 * Revision 1.2  1993/05/03  17:14:24  foster
 * Restructure functions
 *
 * Revision 1.1  1993/05/01  11:35:36  foster
 * Initial revision
 * */

#define MAXPRIO 3
#define MAXLOPRIO 2
#define BLOCKPRIO 0
#define CMDSIZE 20 /* size of command buffer */

/* Scheduling commands */
#define NEW_JOB 1
#define UPGRADE_PRIO 2
#define BLOCK 3 
#define UNBLOCK 4
#define QUANTUM_EXPIRE 5
#define FINISH 6
#define FLUSH 7

/* stati */
#define OK 0
#define TRUE 1
#define FALSE 0
#define BADNOARGS -1 /* Wrong number of arguments */
#define BADARG -2    /* Bad argument (< 0) */
#define MALLOC_ERR -3
#define BADPRIO -4   /* priority < 0 or > MAXPRIO */
#define BADRATIO -5  /* ratio < 0 or > 1 */
#define NO_COMMAND -6 /* No such scheduling command */

extern int get_command(/* int *command, *prio, float *ratio */);
			  /* Get command from stdin. Return 0 on EOF */
extern exit_here(/* int status */); /* Exit program with abs(status) */

extern int enqueue(/* int prio, struct process * job */);
                     /* Put job at end of queue & reschedule */
extern int new_job(/* int prio */);
                     /* allocate process block & enqueue. Return status */
extern int schedule(/* int command, prio, float ratio */);
                     /* Carry out command. Return status */
extern struct process * get_current();
                     /* Get current job. Reschedule, if necessary */
extern int reschedule(/* int prio */);
                        /* If prio higher than current job, reschedule */

extern int put_end(/* int prio, struct process * process */);
		      /* Put process at end of prio queue. Return status */
extern int get_process(/* int prio, float ratio, struct process ** job */);
                 /* get job from prio queue at ratio position. Return status */

struct process
{
    unsigned int pid;
    int priority;
    struct process *next;
};
