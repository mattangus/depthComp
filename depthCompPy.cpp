#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include "depthComp.hpp"

namespace depthComp
{

    namespace p = boost::python;
    namespace np = boost::python::numpy;

    cv::Mat depthComplete(cv::Mat depth, cv::Mat label)
    {
        DepthComp completer;

        depth = completer.preProcess(depth, label, false);
        cv::Mat depthOut = completer.identFillHoles(depth, label, false);
        depthOut = completer.postProcess(depthOut, label, true);
        return depthOut;
    }


    #if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
    #else
        static void init_ar(){
    #endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (libdepthComp) {
        init_ar();

        //initialize converters
        p::to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        p::def("depthComplete", depthComplete);
    }
}