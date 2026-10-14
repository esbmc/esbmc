#ifndef NONDET_H
#define NONDET_H

/* Regression for #7434: the nondet assignment below is on line 29 of this
 * header. Before the fix ESBMC emitted a function_return waypoint naming
 * main.c -- a twelve-line file -- at line 29. README-YAML.md requires a
 * waypoint's file_name to be one of task.input_files, so a waypoint that
 * cannot name one is dropped instead. */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
/* pad */
int __VERIFIER_nondet_int(void);

static int from_header(void)
{
  int t = __VERIFIER_nondet_int();
  return t + 1;
}

#endif
