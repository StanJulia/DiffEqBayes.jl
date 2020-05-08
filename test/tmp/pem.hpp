
// Code generated by stanc f556d0d
#include <stan/model/model_header.hpp>
namespace pem_model_namespace {

template <typename T, typename S>
std::vector<T> resize_to_match__(std::vector<T>& dst, const std::vector<S>& src) {
  dst.resize(src.size());
  return dst;
}

template <typename T>
Eigen::Matrix<T, -1, -1>
resize_to_match__(Eigen::Matrix<T, -1, -1>& dst, const Eigen::Matrix<T, -1, -1>& src) {
  dst.resize(src.rows(), src.cols());
  return dst;
}

template <typename T>
Eigen::Matrix<T, 1, -1>
resize_to_match__(Eigen::Matrix<T, 1, -1>& dst, const Eigen::Matrix<T, 1, -1>& src) {
  dst.resize(src.size());
  return dst;
}

template <typename T>
Eigen::Matrix<T, -1, 1>
resize_to_match__(Eigen::Matrix<T, -1, 1>& dst, const Eigen::Matrix<T, -1, 1>& src) {
  dst.resize(src.size());
  return dst;
}
std::vector<double> to_doubles__(std::initializer_list<double> x) {
  return x;
}

std::vector<stan::math::var> to_vars__(std::initializer_list<stan::math::var> x) {
  return x;
}

inline void validate_positive_index(const char* var_name, const char* expr,
                                    int val) {
  if (val < 1) {
    std::stringstream msg;
    msg << "Found dimension size less than one in simplex declaration"
        << "; variable=" << var_name << "; dimension size expression=" << expr
        << "; expression value=" << val;
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}

inline void validate_unit_vector_index(const char* var_name, const char* expr,
                                       int val) {
  if (val <= 1) {
    std::stringstream msg;
    if (val == 1) {
      msg << "Found dimension size one in unit vector declaration."
          << " One-dimensional unit vector is discrete"
          << " but the target distribution must be continuous."
          << " variable=" << var_name << "; dimension size expression=" << expr;
    } else {
      msg << "Found dimension size less than one in unit vector declaration"
          << "; variable=" << var_name << "; dimension size expression=" << expr
          << "; expression value=" << val;
    }
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}


using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using std::pow;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using stan::model::cons_list;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::nil_index_list;
using namespace stan::math; 

static int current_statement__ = 0;
static const std::vector<string> locations_array__ = {" (found before start of program)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 38, column 4 to column 34)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 39, column 4 to column 16)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 39, column 16 to column 28)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 42, column 4 to column 18)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 43, column 4 to column 22)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 43, column 22 to column 40)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 55, column 4 to column 20)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 56, column 4 to column 88)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 46, column 4 to column 20)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 47, column 4 to column 33)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 48, column 4 to column 31)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 48, column 31 to column 58)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 49, column 4 to column 121)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 51, column 6 to column 56)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 50, column 18 to line 52, column 7)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 50, column 4 to line 52, column 7)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 27, column 4 to column 15)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 28, column 4 to column 19)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 29, column 4 to column 31)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 30, column 4 to column 12)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 31, column 4 to column 15)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 34, column 4 to column 16)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 35, column 4 to column 15)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 3, column 4 to column 44)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 4, column 4 to column 33)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 5, column 4 to column 12)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 11, column 12 to column 36)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 12, column 12 to column 65)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 10, column 15 to line 13, column 9)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 8, column 12 to column 25)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 9, column 12 to column 22)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 7, column 25 to line 10, column 9)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 7, column 8 to line 13, column 9)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 6, column 22 to line 14, column 5)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 6, column 4 to line 14, column 5)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 15, column 4 to column 15)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 2, column 54 to line 16, column 1)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 19, column 2 to column 28)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 20, column 2 to column 111)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 21, column 2 to column 94)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 22, column 2 to column 27)",
                                                      " (in '/Users/rob/.julia/dev/DiffEqBayes/test/tmp/pem.stan', line 18, column 92 to line 23, column 1)"};


template <typename T0__>
int
bin_search(const T0__& x, const int& min_val, const int& max_val,
           std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    int range;
    
    current_statement__ = 24;
    range = (((max_val - min_val) + 1) / 2);
    int mid_pt;
    
    current_statement__ = 25;
    mid_pt = (min_val + range);
    int out;
    
    current_statement__ = 35;
    while (logical_gt(range, 0)) {
      current_statement__ = 33;
      if (logical_eq(x, mid_pt)) {
        current_statement__ = 30;
        out = mid_pt;
        current_statement__ = 31;
        range = 0;
      } else {
        current_statement__ = 27;
        range = ((range + 1) / 2);
        current_statement__ = 28;
        mid_pt = (logical_gt(x, mid_pt) ? (mid_pt + range) : (mid_pt - range));
      }
    }
    current_statement__ = 36;
    return out;
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct bin_search_functor__ {
template <typename T0__>
int
operator()(const T0__& x, const int& min_val, const int& max_val,
           std::ostream* pstream__)  const 
{
return bin_search(x, min_val, max_val, pstream__);
}
};

template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
sho(const T0__& t, const std::vector<T1__>& internal_var___u,
    const std::vector<T2__>& internal_var___p, const std::vector<T3__>& x_r,
    const std::vector<int>& x_i, std::ostream* pstream__) {
  using local_scalar_t__ = typename boost::math::tools::promote_args<T0__,
          T1__,
          T2__,
          T3__>::type;
  const static bool propto__ = true;
  (void) propto__;
  
  try {
    current_statement__ = 38;
    validate_non_negative_index("internal_var___du", "2", 2);
    std::vector<local_scalar_t__> internal_var___du;
    internal_var___du = std::vector<local_scalar_t__>(2, 0);
    
    current_statement__ = 39;
    assign(internal_var___du, cons_list(index_uni(1), nil_index_list()),
      ((internal_var___u[(1 - 1)] * internal_var___p[(1 - 1)]) -
        (internal_var___u[(1 - 1)] * internal_var___u[(2 - 1)])),
      "assigning variable internal_var___du");
    current_statement__ = 40;
    assign(internal_var___du, cons_list(index_uni(2), nil_index_list()),
      ((-3 * internal_var___u[(2 - 1)]) +
        (internal_var___u[(1 - 1)] * internal_var___u[(2 - 1)])),
      "assigning variable internal_var___du");
    current_statement__ = 41;
    return internal_var___du;
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct sho_functor__ {
template <typename T0__, typename T1__, typename T2__, typename T3__>
std::vector<typename boost::math::tools::promote_args<T0__, T1__, T2__,
T3__>::type>
operator()(const T0__& t, const std::vector<T1__>& internal_var___u,
           const std::vector<T2__>& internal_var___p,
           const std::vector<T3__>& x_r, const std::vector<int>& x_i,
           std::ostream* pstream__)  const 
{
return sho(t, internal_var___u, internal_var___p, x_r, x_i, pstream__);
}
};

class pem_model : public model_base_crtp<pem_model> {

 private:
  int pos__;
  std::vector<double> u0;
  int T;
  std::vector<std::vector<double>> internal_var___u;
  double t0;
  std::vector<double> ts;
  std::vector<double> x_r;
  std::vector<int> x_i;
 
 public:
  ~pem_model() { }
  
  std::string model_name() const { return "pem_model"; }
  
  pem_model(stan::io::var_context& context__, unsigned int random_seed__ = 0,
            std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    typedef double local_scalar_t__;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static const char* function__ = "pem_model_namespace::pem_model";
    (void) function__;  // suppress unused var warning
    
    try {
      
      pos__ = 1;
      current_statement__ = 17;
      validate_non_negative_index("u0", "2", 2);
      context__.validate_dims("data initialization","u0","double",
          context__.to_vec(2));
      u0 = std::vector<double>(2, 0);
      
      current_statement__ = 17;
      assign(u0, nil_index_list(), context__.vals_r("u0"),
        "assigning variable u0");
      context__.validate_dims("data initialization","T","int",
          context__.to_vec());
      
      current_statement__ = 18;
      T = context__.vals_i("T")[(1 - 1)];
      current_statement__ = 19;
      validate_non_negative_index("internal_var___u", "T", T);
      current_statement__ = 19;
      validate_non_negative_index("internal_var___u", "1", 1);
      context__.validate_dims("data initialization","internal_var___u",
          "double",context__.to_vec(T, 1));
      internal_var___u = std::vector<std::vector<double>>(T, std::vector<double>(1, 0));
      
      {
        std::vector<local_scalar_t__> internal_var___u_flat__;
        current_statement__ = 19;
        assign(internal_var___u_flat__, nil_index_list(),
          context__.vals_r("internal_var___u"),
          "assigning variable internal_var___u_flat__");
        current_statement__ = 19;
        pos__ = 1;
        current_statement__ = 19;
        for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
          current_statement__ = 19;
          for (size_t sym2__ = 1; sym2__ <= T; ++sym2__) {
            current_statement__ = 19;
            assign(internal_var___u,
              cons_list(index_uni(sym2__),
                cons_list(index_uni(sym1__), nil_index_list())),
              internal_var___u_flat__[(pos__ - 1)],
              "assigning variable internal_var___u");
            current_statement__ = 19;
            pos__ = (pos__ + 1);}}
      }
      context__.validate_dims("data initialization","t0","double",
          context__.to_vec());
      
      current_statement__ = 20;
      t0 = context__.vals_r("t0")[(1 - 1)];
      current_statement__ = 21;
      validate_non_negative_index("ts", "T", T);
      context__.validate_dims("data initialization","ts","double",
          context__.to_vec(T));
      ts = std::vector<double>(T, 0);
      
      current_statement__ = 21;
      assign(ts, nil_index_list(), context__.vals_r("ts"),
        "assigning variable ts");
      current_statement__ = 22;
      validate_non_negative_index("x_r", "0", 0);
      x_r = std::vector<double>(0, 0);
      
      current_statement__ = 23;
      validate_non_negative_index("x_i", "0", 0);
      x_i = std::vector<int>(0, 0);
      
      current_statement__ = 18;
      current_statement__ = 18;
      check_greater_or_equal(function__, "T", T, 1);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = 0U;
    
    try {
      current_statement__ = 1;
      validate_non_negative_index("sigma1", "1", 1);
      num_params_r__ += 1;
      num_params_r__ += 1;
      num_params_r__ += 1;
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
  }
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__, std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    typedef T__ local_scalar_t__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    static const char* function__ = "pem_model_namespace::log_prob";
(void) function__;  // suppress unused var warning

    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    
    try {
      current_statement__ = 1;
      validate_non_negative_index("sigma1", "1", 1);
      Eigen::Matrix<local_scalar_t__, 1, -1> sigma1;
      sigma1 = Eigen::Matrix<local_scalar_t__, 1, -1>(1);
      
      current_statement__ = 1;
      sigma1 = in__.row_vector(1);
      current_statement__ = 1;
      for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
        current_statement__ = 1;
        if (jacobian__) {
          current_statement__ = 1;
          assign(sigma1, cons_list(index_uni(sym1__), nil_index_list()),
            stan::math::lb_constrain(sigma1[(sym1__ - 1)], 0, lp__),
            "assigning variable sigma1");
        } else {
          current_statement__ = 1;
          assign(sigma1, cons_list(index_uni(sym1__), nil_index_list()),
            stan::math::lb_constrain(sigma1[(sym1__ - 1)], 0),
            "assigning variable sigma1");
        }}
      local_scalar_t__ theta1;
      
      current_statement__ = 2;
      theta1 = in__.scalar();
      local_scalar_t__ theta2;
      
      current_statement__ = 3;
      theta2 = in__.scalar();
      current_statement__ = 4;
      validate_non_negative_index("theta", "2", 2);
      std::vector<local_scalar_t__> theta;
      theta = std::vector<local_scalar_t__>(2, 0);
      
      current_statement__ = 5;
      assign(theta, cons_list(index_uni(1), nil_index_list()), theta1,
        "assigning variable theta");
      current_statement__ = 6;
      assign(theta, cons_list(index_uni(2), nil_index_list()), theta2,
        "assigning variable theta");
      {
        current_statement__ = 9;
        validate_non_negative_index("u_hat", "T", T);
        current_statement__ = 9;
        validate_non_negative_index("u_hat", "2", 2);
        std::vector<std::vector<local_scalar_t__>> u_hat;
        u_hat = std::vector<std::vector<local_scalar_t__>>(T, std::vector<local_scalar_t__>(2, 0));
        
        current_statement__ = 10;
        lp_accum__.add(inv_gamma_log<propto__>(sigma1, 3.0, 3.0));
        current_statement__ = 11;
        lp_accum__.add(normal_log<propto__>(theta[(1 - 1)], 1.0, 0.5));
        current_statement__ = 12;
        lp_accum__.add(normal_log<propto__>(theta[(2 - 1)], 1.5, 0.5));
        current_statement__ = 13;
        assign(u_hat, nil_index_list(),
          integrate_ode_rk45(sho_functor__(),
            append_array(
              rvalue(theta, cons_list(index_min_max(1, 1), nil_index_list()),
                "theta"), stan::math::array_builder<double>().add(1.0)
              .array()), t0, ts,
            rvalue(theta, cons_list(index_min_max(2, 2), nil_index_list()),
              "theta"), x_r, x_i, pstream__, 0.001, 1.0e-6, 100000),
          "assigning variable u_hat");
        current_statement__ = 16;
        for (size_t t = 1; t <= T; ++t) {
          current_statement__ = 14;
          lp_accum__.add(
            normal_log<propto__>(
              rvalue(internal_var___u,
                cons_list(index_uni(t),
                  cons_list(index_omni(), nil_index_list())),
                "internal_var___u"), u_hat[(t - 1)][(1 - 1)], sigma1));}
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob() 
    
  template <typename RNG>
  void write_array(RNG& base_rng__, std::vector<double>& params_r__,
                   std::vector<int>& params_i__, std::vector<double>& vars__,
                   bool emit_transformed_parameters__ = true,
                   bool emit_generated_quantities__ = true,
                   std::ostream* pstream__ = 0) const {
    typedef double local_scalar_t__;
    vars__.resize(0);
    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    static const char* function__ = "pem_model_namespace::write_array";
(void) function__;  // suppress unused var warning

    (void) function__;  // suppress unused var warning

    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    stan::math::accumulator<double> lp_accum__;
    
    try {
      current_statement__ = 1;
      validate_non_negative_index("sigma1", "1", 1);
      Eigen::Matrix<double, 1, -1> sigma1;
      sigma1 = Eigen::Matrix<double, 1, -1>(1);
      
      current_statement__ = 1;
      sigma1 = in__.row_vector(1);
      current_statement__ = 1;
      for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
        current_statement__ = 1;
        assign(sigma1, cons_list(index_uni(sym1__), nil_index_list()),
          stan::math::lb_constrain(sigma1[(sym1__ - 1)], 0),
          "assigning variable sigma1");}
      double theta1;
      
      current_statement__ = 2;
      theta1 = in__.scalar();
      double theta2;
      
      current_statement__ = 3;
      theta2 = in__.scalar();
      current_statement__ = 4;
      validate_non_negative_index("theta", "2", 2);
      std::vector<double> theta;
      theta = std::vector<double>(2, 0);
      
      for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
        vars__.push_back(sigma1[(sym1__ - 1)]);}
      vars__.push_back(theta1);
      vars__.push_back(theta2);
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 5;
      assign(theta, cons_list(index_uni(1), nil_index_list()), theta1,
        "assigning variable theta");
      current_statement__ = 6;
      assign(theta, cons_list(index_uni(2), nil_index_list()), theta2,
        "assigning variable theta");
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.push_back(theta[(sym1__ - 1)]);}
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      current_statement__ = 7;
      validate_non_negative_index("u_hat", "T", T);
      current_statement__ = 7;
      validate_non_negative_index("u_hat", "2", 2);
      std::vector<std::vector<double>> u_hat;
      u_hat = std::vector<std::vector<double>>(T, std::vector<double>(2, 0));
      
      current_statement__ = 8;
      assign(u_hat, nil_index_list(),
        integrate_ode_rk45(sho_functor__(), u0, t0, ts, theta, x_r, x_i,
          pstream__, 0.001, 1.0e-6, 100000), "assigning variable u_hat");
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        for (size_t sym2__ = 1; sym2__ <= T; ++sym2__) {
          vars__.push_back(u_hat[(sym2__ - 1)][(sym1__ - 1)]);}}
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array() 
    
  void transform_inits(const stan::io::var_context& context__,
                       std::vector<int>& params_i__,
                       std::vector<double>& vars__, std::ostream* pstream__) const {
    typedef double local_scalar_t__;
    vars__.resize(0);
    vars__.reserve(num_params_r__);
    
    try {
      int pos__;
      
      pos__ = 1;
      current_statement__ = 1;
      validate_non_negative_index("sigma1", "1", 1);
      Eigen::Matrix<double, 1, -1> sigma1;
      sigma1 = Eigen::Matrix<double, 1, -1>(1);
      
      {
        std::vector<local_scalar_t__> sigma1_flat__;
        current_statement__ = 1;
        assign(sigma1_flat__, nil_index_list(), context__.vals_r("sigma1"),
          "assigning variable sigma1_flat__");
        current_statement__ = 1;
        pos__ = 1;
        current_statement__ = 1;
        for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
          current_statement__ = 1;
          assign(sigma1, cons_list(index_uni(sym1__), nil_index_list()),
            sigma1_flat__[(pos__ - 1)], "assigning variable sigma1");
          current_statement__ = 1;
          pos__ = (pos__ + 1);}
      }
      current_statement__ = 1;
      for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
        current_statement__ = 1;
        assign(sigma1, cons_list(index_uni(sym1__), nil_index_list()),
          stan::math::lb_free(sigma1[(sym1__ - 1)], 0),
          "assigning variable sigma1");}
      double theta1;
      
      current_statement__ = 2;
      theta1 = context__.vals_r("theta1")[(1 - 1)];
      double theta2;
      
      current_statement__ = 3;
      theta2 = context__.vals_r("theta2")[(1 - 1)];
      for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
        vars__.push_back(sigma1[(sym1__ - 1)]);}
      vars__.push_back(theta1);
      vars__.push_back(theta2);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits() 
    
  void get_param_names(std::vector<std::string>& names__) const {
    
    names__.resize(0);
    names__.push_back("sigma1");
    names__.push_back("theta1");
    names__.push_back("theta2");
    names__.push_back("theta");
    names__.push_back("u_hat");
    } // get_param_names() 
    
  void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> dims__;
    dims__.push_back(1);
    dimss__.push_back(dims__);
    dims__.resize(0);
    dimss__.push_back(dims__);
    dims__.resize(0);
    dimss__.push_back(dims__);
    dims__.resize(0);
    dims__.push_back(2);
    dimss__.push_back(dims__);
    dims__.resize(0);
    dims__.push_back(T);
    
    dims__.push_back(2);
    dimss__.push_back(dims__);
    dims__.resize(0);
    
    } // get_dims() 
    
  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool emit_transformed_parameters__ = true,
                               bool emit_generated_quantities__ = true) const {
    
    for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
      {
        param_names__.push_back(std::string() + "sigma1" + '.' + std::to_string(sym1__));
      }}
    param_names__.push_back(std::string() + "theta1");
    param_names__.push_back(std::string() + "theta2");
    if (emit_transformed_parameters__) {
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          param_names__.push_back(std::string() + "theta" + '.' + std::to_string(sym1__));
        }}
    }
    
    if (emit_generated_quantities__) {
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          for (size_t sym2__ = 1; sym2__ <= T; ++sym2__) {
            {
              param_names__.push_back(std::string() + "u_hat" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
            }}
        }}
    }
    
    } // constrained_param_names() 
    
  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool emit_transformed_parameters__ = true,
                                 bool emit_generated_quantities__ = true) const {
    
    for (size_t sym1__ = 1; sym1__ <= 1; ++sym1__) {
      {
        param_names__.push_back(std::string() + "sigma1" + '.' + std::to_string(sym1__));
      }}
    param_names__.push_back(std::string() + "theta1");
    param_names__.push_back(std::string() + "theta2");
    if (emit_transformed_parameters__) {
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          param_names__.push_back(std::string() + "theta" + '.' + std::to_string(sym1__));
        }}
    }
    
    if (emit_generated_quantities__) {
      for (size_t sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          for (size_t sym2__ = 1; sym2__ <= T; ++sym2__) {
            {
              param_names__.push_back(std::string() + "u_hat" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
            }}
        }}
    }
    
    } // unconstrained_param_names() 
    
  std::string get_constrained_sizedtypes() const {
    stringstream s__;
    s__ << "[{\"name\":\"sigma1\",\"type\":{\"name\":\"vector\",\"length\":" << 1 << "},\"block\":\"parameters\"},{\"name\":\"theta1\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta\",\"type\":{\"name\":\"array\",\"length\":" << 2 << ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"u_hat\",\"type\":{\"name\":\"array\",\"length\":" << T << ",\"element_type\":{\"name\":\"array\",\"length\":" << 2 << ",\"element_type\":{\"name\":\"real\"}}},\"block\":\"generated_quantities\"}]";
    return s__.str();
    } // get_constrained_sizedtypes() 
    
  std::string get_unconstrained_sizedtypes() const {
    stringstream s__;
    s__ << "[{\"name\":\"sigma1\",\"type\":{\"name\":\"vector\",\"length\":" << 1 << "},\"block\":\"parameters\"},{\"name\":\"theta1\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta2\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"theta\",\"type\":{\"name\":\"array\",\"length\":" << 2 << ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"u_hat\",\"type\":{\"name\":\"array\",\"length\":" << T << ",\"element_type\":{\"name\":\"array\",\"length\":" << 2 << ",\"element_type\":{\"name\":\"real\"}}},\"block\":\"generated_quantities\"}]";
    return s__.str();
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    void write_array(RNG& base_rng__,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool emit_transformed_parameters__ = true,
                     bool emit_generated_quantities__ = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng__, params_r_vec, params_i_vec, vars_vec,
          emit_transformed_parameters__, emit_generated_quantities__, pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }

    template <bool propto__, bool jacobian__, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto__,jacobian__,T_>(vec_params_r, vec_params_i, pstream);
    }

    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }

};
}
typedef pem_model_namespace::pem_model stan_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

#endif


