#include <langapi/mode.h>

// Minimal mode_table definition for unit tests that pull in langapi
// (via gotoprograms -> language_util.h -> new_language -> mode_table).
// Tests using test_goto_factory do not need this; goto_factory.cpp defines it.
// Use as an OBJECT library so the .o is always included in the link.
// Minimal stub: no languages registered.  Tests that use this stub never
// actually call new_language() at runtime; the symbol just needs to exist so
// that langapi/mode.cpp.o links cleanly without requiring clangcfrontend.
const mode_table_et mode_table[] = {LANGAPI_MODE_END};
