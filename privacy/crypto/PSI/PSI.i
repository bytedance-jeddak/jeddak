%module PSI

%{
#define SWIG_FILE_WITH_INIT
#include "include/Defines.h"
#include "include/PsiSender.h"
#include "include/PsiReceiver.h"
#include "include/utils.h"

%}
%include "numpy.i"
%include "std_vector.i"
%include "carrays.i"

%init %{
import_array(); // This is essential. We will get a crash in Python without it.
%}

//%apply (unsigned char* INPLACE_ARRAY2, int DIM1, int DIM2) {(unsigned char *matrixD, int dim1_D, int dim2_D)}

%template(uncharVector) std::vector<unsigned char>;
%array_class(DATA, dataArray)
%array_class(unsigned char, uncharArray)
%array_class(unsigned long long, u64Array)
%apply (unsigned char* IN_ARRAY1, int DIM1) {(unsigned char *row, int size)};

%include "include/Defines.h"
%include "include/PsiSender.h"
%include "include/PsiReceiver.h"
%include "include/utils.h"


//%clear (unsigned char *matrixD, int dim1_D, int dim2_D);
