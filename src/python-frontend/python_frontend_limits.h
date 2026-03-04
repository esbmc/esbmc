// Copyright (C) 2024-2026 Diffblue Ltd and
// University of Manchester, University of Oxford.
// SPDX-License-Identifier: BSD-4-Clause
#ifndef ESBMC_PYTHON_FRONTEND_LIMITS_H
#define ESBMC_PYTHON_FRONTEND_LIMITS_H

// Upper bound for expanding symbolic/implicit sequences to keep model checking
// tractable (e.g., range() materialization, string repetition folding).
constexpr long long kMaxSequenceExpansion = 10000;

#endif // ESBMC_PYTHON_FRONTEND_LIMITS_H
