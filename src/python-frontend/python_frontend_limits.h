#ifndef ESBMC_PYTHON_FRONTEND_LIMITS_H
#define ESBMC_PYTHON_FRONTEND_LIMITS_H

// Upper bound for expanding symbolic/implicit sequences
// (e.g., range() materialization, string repetition folding).
constexpr long long kMaxSequenceExpansion = 10000;

#endif // ESBMC_PYTHON_FRONTEND_LIMITS_H
