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
// RvalueCvt: Converter to make an Rvalue of Output in temporary storage
// LvalueCvt: Conversion from InPython to Output pointer. Necessary for
//            _both_ rvalue and lvalue conversion.
template <typename Output, typename InPython, bool AllowNone, bool Rvalue,
         bool Lvalue, typename RvalueCvt, typename LvalueCvt>
struct esbmc_python_cvt
{
  static RvalueCvt rvalue_cvt;
  static LvalueCvt lvalue_cvt;

    esbmc_python_cvt(RvalueCvt _rvalue_cvt, LvalueCvt _lvalue_cvt)
    {
      using namespace boost::python;
      rvalue_cvt(_rvalue_cvt);
      lvalue_cvt(_lvalue_cvt);

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

      // InPython needs to be a pointer so that it's held in a pointer_holder.
      static_assert(boost::is_pointer<InPython>::value, "InPython must be considered a pointer");

      // Slightly dirtily extricate the m_p field. Don't call pointer_holder
      // holds because that's private. Ugh.
      void *in_python = const_cast<void*>(reinterpret_cast<const void *>(foo->get()));

      // Allow lvalue conversion to perhaps mush this around, perhaps it
      // changes the pointer to point inside something? Either way: it's not
      // permitted to create new storage.
      return lvalue_cvt(in_python);
    }

    static void cons(PyObject *src, boost::python::converter::rvalue_from_python_stage1_data *stage1)
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
      rvalue_cvt(stage1->convertible, obj_store);

      // Let rvalue holder know that needs deconstructing please
      store->stage1.convertible = obj_store;
      return;
    }
};
