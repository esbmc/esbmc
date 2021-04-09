/*******************************************************************\

Module: class to store information of each declarations in Solidity AST???

Author: Kunjian Song

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECLARATION_H
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECLARATION_H

#include <cassert>
#include <util_solidity/solidity_symbol.h>

class solidity_declarationt : public solidity_exprt
{
public:
    solidity_declarationt()
    {
    }

    void to_symbol(solidity_symbolt &symbol) const
    {
        // TODO: cf. src/ansi-c/ansi_c_declaration.h
        assert(!"come back and continue - solidity declarationt::to_symbol");
    }
};

#endif
