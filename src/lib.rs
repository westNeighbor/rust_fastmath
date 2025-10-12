use pyo3::prelude::*;
use pyo3::types::PyDict;
use meval::{Expr, Context};
use std::collections::HashMap;
use std::str::FromStr;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ==================================================================================
// --- ODE Solvers Helper (Used by RK4 and RKF45) ---
// ==================================================================================

struct ODESystem {
    var_names: Vec<String>,
    parsed_exprs: Vec<Expr>,
    num_vars: usize,
    base_context: Context<'static>, 
}

impl ODESystem {
    fn new(exprs: Vec<(String, String)>, params: &Bound<'_, PyDict>) -> PyResult<Self> {
        let var_names: Vec<String> = exprs.iter().map(|(name, _)| name.clone()).collect();
        let num_vars = var_names.len();

        let parsed_exprs: Vec<Expr> = exprs
            .iter()
            .map(|(name, expr_str)| {
                Expr::from_str(expr_str).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid expr for {}: {}", name, err
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        let mut base_context = Context::new();
        for (k, v) in params.iter() {
            let key: &'static str = Box::leak(k.extract::<String>()?.into_boxed_str());
            let val: f64 = v.extract()?;
            base_context.var(key, val);
        }

        Ok(ODESystem {
            var_names,
            parsed_exprs,
            num_vars,
            base_context,
        })
    }

    fn eval_derivs(&self, current_state: &[f64], derivs: &mut [f64]) {
        let mut ctx = self.base_context.clone();
        
        for i in 0..self.num_vars {
            ctx.var(self.var_names[i].clone(), current_state[i]);
        }
        
        for i in 0..self.num_vars {
            derivs[i] = self.parsed_exprs[i].eval_with_context(ctx.clone()).unwrap_or(0.0);
        }
    }
}

// ==================================================================================
// --- 1. Fixed-Step Runge-Kutta 4 (RK4) Solver ---
// ==================================================================================

#[pyfunction]
fn solve_expr_ode_rk4<'py>(
    py: Python<'py>,
    exprs: Vec<(String, String)>,
    params: &Bound<'_, PyDict>,
    y0: &Bound<'_, PyDict>,
    t_span: (f64, f64),
    dt: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    
    let sys = ODESystem::new(exprs, params)?;
    let num_vars = sys.num_vars;
    let (t0, t1) = t_span;

    if dt <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("dt must be positive."));
    }

    let n_steps = ((t1 - t0) / dt).floor() as usize;
    let steps_to_run = n_steps + 1;

    let mut state: Vec<f64> = vec![0.0; num_vars];
    let y0_map: HashMap<String, f64> = y0.extract()?;
    for (i, name) in sys.var_names.iter().enumerate() {
        state[i] = *y0_map.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Initial value for '{}' not found in y0", name))
        })?;
    }
    
    let mut results_flat = Vec::with_capacity(steps_to_run * num_vars);

    let mut k1 = vec![0.0; num_vars];
    let mut k2 = vec![0.0; num_vars];
    let mut k3 = vec![0.0; num_vars];
    let mut k4 = vec![0.0; num_vars];
    let mut temp_state = vec![0.0; num_vars];
    
    let mut _t = t0; 

    results_flat.extend_from_slice(&state);

    // Main RK4 Loop
    for _ in 0..n_steps {
        sys.eval_derivs(&state, &mut k1);

        for i in 0..num_vars { temp_state[i] = state[i] + dt * k1[i] / 2.0; }
        sys.eval_derivs(&temp_state, &mut k2);

        for i in 0..num_vars { temp_state[i] = state[i] + dt * k2[i] / 2.0; }
        sys.eval_derivs(&temp_state, &mut k3);

        for i in 0..num_vars { temp_state[i] = state[i] + dt * k3[i]; }
        sys.eval_derivs(&temp_state, &mut k4);

        for i in 0..num_vars {
            state[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        _t += dt; 

        results_flat.extend_from_slice(&state);
    }
    
    // Post-Loop Time Calculation
    let t_vals: Vec<f64> = (0..steps_to_run)
        .map(|i| t0 + i as f64 * dt)
        .collect();

    let final_steps = t_vals.len();

    let t_array = t_vals.into_pyarray_bound(py);
    
    let y_array = results_flat
        .into_pyarray_bound(py)
        .reshape((final_steps, num_vars))?; 
        
    Ok((t_array, y_array))
}

// ==================================================================================
// --- 2. Adaptive Runge-Kutta-Fehlberg 4(5) Solver (RKF45) ---
// ==================================================================================

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_expr_ode_adaptive<'py>(
    py: Python<'py>,
    exprs: Vec<(String, String)>,
    params: &Bound<'_, PyDict>,
    y0: &Bound<'_, PyDict>,
    t_span: (f64, f64),
    mut dt: f64,
    tol: f64,
    dt_min: f64,
    dt_max: f64,
    safety_factor: f64, // larger for better performance, smaller for smoother, suggested 0.84, range 0.7-1.0
    max_growth_factor: f64, // to prevent too large jumps in dt, suggested 2.0, range 1.2-5.0
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    
    let sys = ODESystem::new(exprs, params)?;
    let num_vars = sys.num_vars;
    let (t0, t1) = t_span;

    let mut y: Vec<f64> = vec![0.0; num_vars];
    let y0_map: HashMap<String, f64> = y0.extract()?;
    for (i, name) in sys.var_names.iter().enumerate() {
        y[i] = *y0_map.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Initial value for '{}' not found in y0", name))
        })?;
    }

    let mut t = t0;

    let mut t_vals: Vec<f64> = vec![t];
    let mut y_vals_flat: Vec<f64> = y.clone(); 

    let mut k = vec![0.0; 6 * num_vars];
    let mut y4 = vec![0.0; num_vars];
    let mut y5 = vec![0.0; num_vars];
    let mut temp_y = vec![0.0; num_vars]; 

    const T_EPSILON: f64 = 1e-12; 

    while t < t1 - T_EPSILON {
        let dt_trial = (t1 - t).min(dt); 

        // --- Calculate k-values (using dt_trial) ---
        // k[0]
        sys.eval_derivs(&y, &mut k[0..num_vars]);

        // k[1]
        for i in 0..num_vars { temp_y[i] = y[i] + dt_trial * (k[i] / 4.0); }
        sys.eval_derivs(&temp_y, &mut k[num_vars..2*num_vars]);

        // k[2]
        for i in 0..num_vars { temp_y[i] = y[i] + dt_trial * (3.0*k[i]/32.0 + 9.0*k[num_vars+i]/32.0); }
        sys.eval_derivs(&temp_y, &mut k[2*num_vars..3*num_vars]);

        // k[3]
        for i in 0..num_vars { 
            temp_y[i] = y[i] + dt_trial * (1932.0*k[i]/2197.0 - 7200.0*k[num_vars+i]/2197.0 + 7296.0*k[2*num_vars+i]/2197.0); 
        }
        sys.eval_derivs(&temp_y, &mut k[3*num_vars..4*num_vars]);

        // k[4]
        for i in 0..num_vars { 
            temp_y[i] = y[i] + dt_trial * (439.0*k[i]/216.0 - 8.0*k[num_vars+i] + 3680.0*k[2*num_vars+i]/513.0 - 845.0*k[3*num_vars+i]/4104.0); 
        }
        sys.eval_derivs(&temp_y, &mut k[4*num_vars..5*num_vars]);

        // k[5]
        for i in 0..num_vars { 
            temp_y[i] = y[i] + dt_trial * (-8.0*k[i]/27.0 + 2.0*k[num_vars+i] - 3544.0*k[2*num_vars+i]/2565.0 + 1859.0*k[3*num_vars+i]/4104.0 - 11.0*k[4*num_vars+i]/40.0); 
        }
        sys.eval_derivs(&temp_y, &mut k[5*num_vars..6*num_vars]);

        // y4 estimate (4th order)
        for i in 0..num_vars {
            y4[i] = y[i] + dt_trial * (
                25.0*k[i]/216.0 + 
                1408.0*k[2*num_vars+i]/2565.0 + 
                2197.0*k[3*num_vars+i]/4104.0 - 
                k[4*num_vars+i]/5.0
            );
        }

        // y5 estimate (5th order - used for next state if accepted)
        for i in 0..num_vars {
            y5[i] = y[i] + dt_trial * (
                16.0*k[i]/135.0 + 
                6656.0*k[2*num_vars+i]/12825.0 + 
                28561.0*k[3*num_vars+i]/56430.0 - 
                9.0*k[4*num_vars+i]/50.0 + 
                2.0*k[5*num_vars+i]/55.0
            );
        }

        // --- Error estimate and step control ---
        let err = y5.iter().zip(y4.iter())
            .map(|(v5, v4)| (v5 - v4).abs())
            .fold(0.0f64, f64::max);

        if err < tol {
            // Step ACCEPTED
            t += dt_trial;
            y.copy_from_slice(&y5);
            
            // Record state
            t_vals.push(t);
            y_vals_flat.extend_from_slice(&y);
        }
        
        // Adaptive step size calculation: s = safety_factor * (tol * dt / (2*err))^0.25
        let s = if err < 1e-15 {
            max_growth_factor
        } else {
            safety_factor * ((tol * dt_trial) / (2.0 * err)).powf(0.25)
        };
        
        // Apply max growth factor explicitly
        let s_capped = s.min(max_growth_factor); 
        
        // Set the new dt for the next iteration
        let dt_new = (s_capped * dt_trial).max(dt_min).min(dt_max);

        // Stiffness check
        if t < t1 && dt_new == dt_min && err > tol {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Step size reduced to minimum but tolerance still not met. System may be stiff."
            ));
        }
        
        dt = dt_new;
    }

    let num_recorded_steps = t_vals.len();
    let t_array = t_vals.into_pyarray_bound(py);
    
    let y_array = y_vals_flat
        .into_pyarray_bound(py)
        .reshape((num_recorded_steps, num_vars))?;

    Ok((t_array, y_array))
}


#[pymodule]
fn rust_fastmath(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add your functions here, for example:
    m.add_function(wrap_pyfunction!(solve_expr_ode_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(solve_expr_ode_adaptive, m)?)?;
    Ok(())
}
