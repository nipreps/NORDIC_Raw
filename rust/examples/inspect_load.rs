//! Diagnostic: load a NIfTI with the same code path as the runner and
//! print min/max/mean and a NaN count. Used to debug scl_slope handling.
//!
//! cargo run --release --example inspect_load -- <path-to-nifti>

use std::env;

use nordic_rs::io::load_4d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: inspect_load <path-to-nifti>");
        std::process::exit(2);
    }
    let path = std::path::PathBuf::from(&args[1]);
    let v = load_4d(&path)?;
    let arr = &v.data;

    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut nz = 0usize;
    let mut z = 0usize;
    let mut pmin = f32::INFINITY;
    let mut pmax = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for &v in arr.iter() {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        if v.is_infinite() {
            inf += 1;
            continue;
        }
        if v == 0.0 {
            z += 1;
        } else {
            nz += 1;
        }
        if v < pmin {
            pmin = v;
        }
        if v > pmax {
            pmax = v;
        }
        sum += v as f64;
        n += 1;
    }
    println!("path: {}", path.display());
    println!("shape: {:?}", arr.shape());
    println!("scl_slope (from header): {:.6}", v.header.scl_slope);
    println!("scl_inter (from header): {:.6}", v.header.scl_inter);
    println!("datatype: {:?}", v.header.data_type());
    println!("min={:.4}  max={:.4}  mean={:.4}", pmin, pmax, sum / n.max(1) as f64);
    println!("#nan={}  #inf={}  #zero={}  #nonzero={}", nan, inf, z, nz);
    Ok(())
}
