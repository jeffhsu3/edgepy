# language = c++
# distutils: sources = ../src/Rectangle.cpp

cdef extern from "glm.h" namespace "glm"
    cdef cppclass glm_levenberg:
        glm_levenberg(int, int, double, int, double)
        glm_levenberg()
        int fit
