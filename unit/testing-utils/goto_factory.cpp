#include "goto_factory.h"

#include <cstdio>
#include <fstream>
#include <langapi/mode.h>
#include <langapi/language_ui.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_skip.h>
#include <util/cmdline.h>


const mode_table_et mode_table[] = {
  LANGAPI_HAVE_MODE_CLANG_C,
  LANGAPI_HAVE_MODE_CLANG_CPP,
  LANGAPI_HAVE_MODE_END
}; // This is the simplest way to have this


goto_functionst goto_factory::get_goto_functions(std::istream &c_file) {
    goto_functionst goto_functions;
    /*
     * 1. Create an tmp file from istream
     * 2. Parse the file using clang-frontend
     * 3. Return the result
     */

    // Create tmp file
    std::string filename("tmp.c"); // TODO: Make this unique and add support for CPP
    std::ofstream output(filename); // Change this for C++
    if(!output.good()) {
        perror("Could not create output C file\n");
        exit(1);
    }

    while(c_file.good()) {
        std::string line;
        c_file >> line;
        output << line;
        output << " ";
    }

    output.close();

    cmdlinet cmdline;
    optionst opts;
    cmdline.args.push_back(filename);
    config.ansi_c.set_16();
    opts.cmdline(cmdline);
    opts.set_option("floatbv", true);
    opts.set_option("context-bound", -1);
    opts.set_option("deadlock-check", false);
    opts.set_option("no-bounds-check", true);
    config.options = opts;
    language_uit lui(cmdline);
    lui.context.clear();
    lui.set_verbosity(8);
    if(lui.parse()) return goto_functions; // TODO: This can be used to add testcases for frontend
    if(lui.typecheck()) return goto_functions;
    if(lui.final()) return goto_functions;
    lui.clear_parse();

    migrate_namespace_lookup = new namespacet(lui.context);    

    goto_convert(lui.context, opts, goto_functions, lui.ui_message_handler);

    namespacet ns(lui.context);
    goto_check(ns, opts, goto_functions);
     // remove skips
    remove_skip(goto_functions);

    // remove unreachable code
    Forall_goto_functions(f_it, goto_functions)
      remove_unreachable(f_it->second.body);

    // remove skips
    remove_skip(goto_functions);

    // recalculate numbers, etc.
    goto_functions.update();

    // add loop ids
    goto_functions.compute_loop_numbers();
    return goto_functions;
}

ui_message_handlert goto_factory::get_message_handlert() {
  cmdlinet cmdline;
  language_uit lui(cmdline);
  lui.set_verbosity(8);
  return lui.ui_message_handler;
}