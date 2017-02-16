/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "dstring.h"

#ifdef WITH_PYTHON
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/operators.hpp>

void
build_dstring_python_class(void)
{
  using namespace boost::python;
  using boost::python::self_ns::self;

  // Not much needed from this but the string and constructors. Return c_str
  // as the as_string operator, because otherwise I have to think about
  // reference ownership, wheras a const char * is obviously copy-by-value.
  class_<dstring>("irep_idt", init<const char *>())
    .def(init<const std::string &>())
    .def("as_string", &dstring::c_str)
    .def("get_no", &dstring::get_no)
    .def(self == self)
    .def(self != self)
    .def(self < self);

  class_<std::vector<dstring> >("irep_idt_vec")
          .def(vector_indexing_suite<std::vector<dstring> >());

  class_<string_wrapper>("string_wrapper")
    .def_readwrite("the_string", &string_wrapper::the_string);
}
#endif
