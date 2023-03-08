/// \file
/// Interval Analysis

#ifndef CPROVER_ANALYSES_INTERVAL_ANALYSIS_H
#define CPROVER_ANALYSES_INTERVAL_ANALYSIS_H

#include <goto-programs/goto_functions.h>

void interval_analysis(goto_functionst &goto_functions, const namespacet &ns, bool print_intervals = false);

#endif // CPROVER_ANALYSES_INTERVAL_ANALYSIS_H
