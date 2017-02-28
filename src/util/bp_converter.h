#include <boost/python/operators.hpp>
#include <boost/python/object/find_instance.hpp>

// A generalised template for generating from-python boost.python conversion
// for our various classes. This is designed to have a series of configurable
// options, including:
//  * Whether from-None conversion is permitted
//  * rvalue or lvalue (or both)
//  * code to massage the rvalue/lvalue conversion as appropriate

// docu-format later, but:
// Output: class that this object will enter C++ as
// InPython: type held in python pointer holder, i.e. shared_ptr of Output
// AllowNone: Should this allow PyNone to be constructed as an Output?
// Rvalue: Enable rvalue converter
// Lvalue: Enable lvalue converter
// ValueCvt: Converter functions to make rvalue / lvalue
template <typename Output, typename InPython, bool AllowNone, bool Rvalue,
         bool Lvalue, typename ValueCvt>
struct esbmc_python_cvt
{
    esbmc_python_cvt()
    {
      using namespace boost::python;

      // Lvalue converter
      if (Lvalue) {
        converter::registry::insert(&convertible, type_id<Output>(),
                  &converter::expected_from_python_type_direct<Output>::get_pytype
                  );
      }
      // We appear to need an rvalue converter for transmografying None to
      // containers.
      if (Rvalue) {
        converter::registry::insert(&convertible, &cons, type_id<Output>(),
            &converter::expected_from_python_type_direct<Output>::get_pytype);
      }
    }

 private:
    // This is called by both rvalue and lvalue converters.
    static void* convertible(PyObject* p)
    {
      using namespace boost::python;

      if (p == Py_None) {
        if (AllowNone)
          return p;
        else
          return NULL;
      }

      objects::instance<> *inst =
        reinterpret_cast<objects::instance<>*>(p);
      (void)inst; // For debug / inspection

      // Scatter consts around to ensure that the get() below doesn't trigger
      // detachment.
      const InPython *foo =
        reinterpret_cast<InPython*>(
            objects::find_instance_impl(p, boost::python::type_id<InPython>()));

      // Find object instance may fail
      if (!foo)
        return NULL;

      // Allow lvalue conversion to perhaps mush this around, perhaps it
      // changes the pointer to point inside something? Either way: it's not
      // permitted to create new storage.
      return ValueCvt::lvalue_cvt(foo);
    }

    static void cons(PyObject *src __attribute__((unused)), boost::python::converter::rvalue_from_python_stage1_data *stage1)
    {
      using namespace boost::python;
      // We're a non-reference non-ptr piece of data; therefore we get created
      // as part of arg_rvalue_from_python, and get an associated bit of
      // storage.
      converter::rvalue_from_python_data<Output> *store =
        reinterpret_cast<converter::rvalue_from_python_data<Output>*>(stage1);

      Output *obj_store = reinterpret_cast<Output *>(&store->storage.bytes);

      assert ((src != Py_None || AllowNone) && "rvalue converter handed PyNone when PyNone is disallowed");

      // Pass storage and from-ptr to rvalue converter, which should in-place
      // initialize the storage.
      const InPython *foo = reinterpret_cast<const InPython*>(stage1->convertible);
      void *res_storage = ValueCvt::rvalue_cvt(foo, obj_store);

      // If rvalue converter used the provided storage, this will also cause
      // the constructed object to be deleted.
      store->stage1.convertible = res_storage;
      return;
    }
};
