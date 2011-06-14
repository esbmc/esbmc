#include "cpp.h"

#include <string.h>

int
handle_hooked_header(usch *name)
{

	fprintf(stderr, "Including file %s\n", name);
	return 0;
}
