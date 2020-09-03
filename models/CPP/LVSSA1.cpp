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


std::vector<int> LV(py::array_t<double> rates, 
                            py::array_t<int> init_molecules, double_t tsart, double_t tend){
  // set fixed elements:
  auto c = rates.unchecked<1>();
  auto x0 = init_molecules.unchecked<1>();
  int N = 2;
  int D = 3;
  
  //double tsart =0.0;
  double t = tsart;
  //double tend = 50.0;
  std::vector<double> current_haz={0,0,0} ;
  std::vector<int> x ={0,0};  
  for (ssize_t i = 0; i < 2; i++){
    x[i] = x0(i);
  }

  Eigen::MatrixXi stoic_mat(N, D);
  stoic_mat(0,0) = 1;
  stoic_mat(0,1) = -1;
  stoic_mat(0,2) = 0;
  stoic_mat(1,0) = 0;
  stoic_mat(1,1) = 1;
  stoic_mat(1,2) = -1;
  std::random_device rd;  
  std::mt19937 gen(rd()); 

  
  while (t<tend)
  {
    current_haz[0] = x[0]*c(0) ;
    current_haz[1] = x[0]*c(1)*x[1];
    current_haz[2] = x[1]*c(2);
    
    double tot_hazard = 0.0;
    for (auto& n : current_haz){
        tot_hazard += n;
        }
    for(unsigned i=0;i<current_haz.size();i++){
	    current_haz[i] = current_haz[i]/tot_hazard;}

    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::discrete_distribution<int> d(current_haz.begin(), current_haz.end());
    double random = u(gen);
    double dt = -log(random) / tot_hazard;
    t = t + dt;
    
    int j = d(gen);
    
    //std::vector<int> udate_col = stoic_mat.col(j);

    for(int i=0;i<x.size();i++){
	    x[i] = x[i] + stoic_mat(i,j);}

    //for(unsigned i=0;i<current_haz.size();i++){
	 //   current_haz.pop_back();}


  }

  return x;
}
PYBIND11_PLUGIN(lvssa) {
    pybind11::module m("lvssa", "auto-compiled c++ extension");
    m.def("LV", &LV, py::return_value_policy::take_ownership);
    return m.ptr();
}
