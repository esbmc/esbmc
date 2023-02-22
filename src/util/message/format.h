/*******************************************************************\

Module: Helper module for formatting text
Author: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#pragma once
#include <string>
#include <util/compiler_defs.h>

CC_DIAGNOSTIC_PUSH()
// Clang has no support for nonnull-compare warning
#ifdef __GNUC__
#ifndef __clang__ // For some reason clang also defines GNUC :)
#pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif
#pragma GCC diagnostic ignored "-Waddress"
#endif

#include <fmt/format.h>
CC_DIAGNOSTIC_POP()

#define ESBMC_FORMATS_START_ASSERTION

// For more in-depth for how this specialization work look at
// https://fmt.dev/latest/api.html#format-api

#include <util/message/formats/bigint.h>
#include <util/message/formats/irep_idt.h>
#include <util/message/formats/irept.h>
#include <util/message/formats/exprt.h>
#include <util/message/formats/typet.h>
#include <util/message/formats/locationt.h>
#include <util/message/formats/type2t.h>
#include <util/message/formats/expr2t.h>
#include <util/message/formats/symbol.h>
#include <util/message/formats/side_efect_expr_function_callt.h>

#undef ESBMC_FORMATS_START_ASSERTION
