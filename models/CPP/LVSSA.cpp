//#include <boost/math/distributions/negative_binomial.hpp>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <stdio.h>      /* printf */
#include <math.h>       /* lgamma */
#include <random>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <cmath>
namespace py = pybind11;
/*-----------------------------------------------------------
Function ICUsim:
1) approximates on intervals dt the system of equation of a SEEIIR model 
 with time varying transmission, 
2) computes, by deterministic convolution, the weekly number of 
 ICU admissions
3) simulates the dataset with negative binomial noise

 takes as a input: THETA a vector of size 6 with elements;
 - beta   : transmission rate
 - pi    : the proportion of initially immune;
 - iota  : the proportion of initially infected/infectious;
 - kappa : the factor for school holidays;
 - pIC   : the probability of IC admissions given infection;
 - eta   : the overdispersion parameter.
 
 gives as output:
 - yIC   : an integer vector (usually of size 33) 
 for the weekly number of IC admissions;
 
 ------------------------------------------------------------- */
//Eigen::MatrixXi

 void LV(py::array_t<double> beta, py::array_t<double> lamda, py::array_t<double> i1,
py::array_t<double> i2, py::array_t<double> e1, py::array_t<double> e2,
py::array_t<double> r, py::array_t<double> s, py::array_t<double> mix_mat, 
py::array_t<double> m_prelock, py::array_t<double> m_postlock, double r0_init, double r0star, 
int Tlock, double gamma, double sigma, double dt, int T, int A){
  // set fixed elements:
  auto beta_ = beta.mutable_unchecked<1>(); 
  auto lamda_ = lamda.mutable_unchecked<2>();
  auto i1_ = i1.mutable_unchecked<2>(); 
  auto i2_ = i2.mutable_unchecked<2>(); 
  auto e1_ = e1.mutable_unchecked<2>(); 
  auto e2_ = e2.mutable_unchecked<2>(); 
  auto r_ = r.mutable_unchecked<2>(); 
  auto s_ = s.mutable_unchecked<2>(); 
  auto mix_mat_ = mix_mat.mutable_unchecked<3>(); 
  auto m_prelock_ = m_prelock.mutable_unchecked<2>(); 
  auto m_postlock_ = m_postlock.mutable_unchecked<2>(); 
   
    for (ssize_t i = 0; i < s_.shape(0); i++)
        for (ssize_t a = 0; a < s.shape(1); a++){





        }
            
}
PYBIND11_PLUGIN(lvssa) {
    pybind11::module m("lvssa", "auto-compiled c++ extension");
    m.def("LV", &LV, py::arg().noconvert(), py::arg().noconvert(), py::arg());//py::return_value_policy::take_ownership);
    return m.ptr();
}
