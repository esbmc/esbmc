#ifndef CPROVER_GOTO_SYMEX_XML_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_XML_GOTO_TRACE_H

#include <goto-symex/goto_trace.h>
#include <util/xml.h>

void convert(const namespacet &ns, const goto_tracet &goto_trace, xmlt &xml);

#endif
