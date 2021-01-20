#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>      /* printf */
#include <math.h>       /* lgamma */
#include <array>
#include <boost/numeric/odeint.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>

#define PI 3.14159265

namespace py = pybind11;
using namespace boost::numeric::odeint;

typedef runge_kutta_dopri5< double > stepper_type;

class ou_ode_class
{
  std::vector<double> m_params;
  double m_T;
  public:
  ou_ode_class( std::vector<double> params, double T ) : m_params(params), m_T(T) { }
  
    void operator()( const double x , double &dxdt , const double t)
    {
      
      double expansion = 0.0;
      for(int i=0; i<m_params.size()-1;++i)
        { double _2i = static_cast<double>(2.0*i);
          double A1 = ( ((_2i)-1.0)*PI);
          double A2 = (2.0*m_T);
          double mp = m_params[i];
          expansion += mp * sqrt(2.0/m_T)*cos((A1/A2)*t);
        }
      dxdt = (1.0*(0.0-x)) + (m_params.back()*expansion);
    }
};

struct push_back_state_and_time
{
    std::vector< double >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< double > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const double &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};
std::vector<double> DailyBrownianBetas(py::array_t<double> beta_params, int ou_weeks, 
int seventh_breakpoint, int last_breakpoint, int lastm1_breakpoint, int grid_space){
  // set fixed elements:
  auto beta_par = beta_params.unchecked<1>(); 
  std::vector<double> params;
  for(ssize_t i=0;i<beta_par.shape(0);++i)
    {
        params.push_back(beta_par(i)); 
    }

  std::vector<double> beta;
  double x = 0.0;
  std::vector< double > x_vec;
  std::vector< double > t_vec;
  /* ou_weeks---This should be number of weeks when beta is varying + 1,
  to account the initial value i.e beta=0 or the week after lockdown
  upto the nearest Tuesday to end date of analysis*/
  std::array<double, 17> ou_vec;

  
  double t_end = static_cast<double>(ou_weeks);
  ou_ode_class ou_ode(params, t_end);
  integrate_adaptive( make_controlled( 1E-6 , 1E-6 , stepper_type() ) ,
                      ou_ode , x , 0.0 , t_end, 0.1 , push_back_state_and_time(x_vec, t_vec));

 
  for(int j=0; j<ou_weeks; ++j)
    {
      int i=0;
      while(i<t_vec.size())
        {
          if(abs(i-j)<1E-20)
            {
              ou_vec[j] = x_vec[j];
            }
          i++;
        }
    }

  int days_in_week = 7*grid_space;
  int last_const_beta_days = (last_breakpoint + 1) - lastm1_breakpoint;
// `<=`---This is to copy the last week (16-th) to rest of the days until analysis ends
  for(int i=0; i<=ou_weeks;i++)
    {
    if(i==0)
      {
        for(int j=0; j<seventh_breakpoint;++j)
          {
            beta.push_back(ou_vec[i]);
          }
      }
// this is to copy the last week 16-th to rest of the days, otherwise use `ou_weeks-1`
    else if(i==ou_weeks)
      {
        for(int k=0; k<last_const_beta_days;++k)
          {
            beta.push_back(ou_vec[i-1]);
          }
      }    
    else
      {
        for(int l=0; l<days_in_week;++l)
          {
            beta.push_back(ou_vec[i]);
          }
      }
    }

  return beta;
}
PYBIND11_PLUGIN(daily_betas) {
    pybind11::module m("daily_betas", "auto-compiled c++ extension");
    m.def("DailyBrownianBetas", &DailyBrownianBetas, py::return_value_policy::take_ownership);
    return m.ptr();
}
