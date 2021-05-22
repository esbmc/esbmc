// Helper for tests that uses goto
#pragma once
#include <istream>
#include <goto-programs/goto_functions.h>
#include <util/ui_message.h>

/**
 * @brief This class parses C inputs
 * and generates goto-functions
 * 
 */
class goto_factory {
public:
    /**
     * @brief Get the goto functions object
     * 
     * @param c_inputstream input stream containing the C program
     * @return goto_functionst of the parsed object
     */
    static goto_functionst get_goto_functions(std::istream &c_inputstream);
    static ui_message_handlert get_message_handlert();
};