/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_OBJECT_NUMBERING_H
#define CPROVER_POINTER_ANALYSIS_OBJECT_NUMBERING_H

#include <expr.h>
#include <hash_cont.h>
#include <irep2.h>
#include <numbering.h>

typedef hash_numbering<expr2tc, irep2_hash> object_numberingt;

#endif
