#include <goto-programs/goto_functions.h>
/*
    To deal with the issue where the the function_call instruction
    and the body definition are not in the same file. 
    ESBMC only shows the failed claim location in the body, 
    which is not relevant for vulnerability repair and bug fixing.
*/
void goto_location(goto_functionst &goto_functions, std::string src_fl);