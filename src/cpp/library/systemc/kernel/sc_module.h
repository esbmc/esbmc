
/***************************************************************
*     sc_module.h
***************************************************************/
#ifndef SC_MODULE_H
#define SC_MODULE_H

class sc_module {
   
    public :

        
        sc_module() {}

        sc_module( const char * label ) {}
        
        ~sc_module() {}  
        
};


/* MACROS */
/* The macros below define the base structs of a SystemC program */

#define SC_MODULE(module_name) struct module_name : public sc_module

#define SC_CTOR(module_name) module_name(const char * label)

/* END MACROS */

#endif
