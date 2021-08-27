%module iknpOTe

%{
#define SWIG_FILE_WITH_INIT
#include "include/Defines.h"
#include "include/utils.h"
#include "include/iknpOTeReceiver.h"
#include "include/iknpOTeSender.h"
%}

%include "numpy.i"
%include "carrays.i"
%include "std_vector.i"
%init %{
import_array(); // This is essential. We will get a crash in Python without it.
%}

%apply (unsigned long long* IN_ARRAY1, int DIM1) {(unsigned long long *numpy_arr, int size)};
%template(uncharVector) std::vector<unsigned char>;
%template(u64Vector) std::vector<unsigned long long>;
%include "include/Defines.h"
%include "include/iknpOTeReceiver.h"
%include "include/iknpOTeSender.h"
%include "include/utils.h"






