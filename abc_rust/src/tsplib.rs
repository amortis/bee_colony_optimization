//! Minimal TSPLIB reader: `EUC_2D` instances with `NODE_COORD_SECTION`.

use crate::tsp::DistanceMatrix;
use std::fs::read_to_string;
use std::path::Path;

#[derive(Debug)]
pub enum TsplibError {
    Io(std::io::Error),
    Parse(&'static str),
}

impl std::fmt::Display for TsplibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TsplibError::Io(e) => write!(f, "{e}"),
            TsplibError::Parse(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for TsplibError {}

impl From<std::io::Error> for TsplibError {
    fn from(e: std::io::Error) -> Self {
        TsplibError::Io(e)
    }
}

fn nint(x: f64) -> f64 {
    x.round()
}

fn euc_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let xd = x1 - x2;
    let yd = y1 - y2;
    nint((xd * xd + yd * yd).sqrt())
}

/// Load symmetric TSP with Euclidean distances (TSPLIB `EUC_2D`).
pub fn load_euc_2d(path: &Path) -> Result<DistanceMatrix, TsplibError> {
    let text = read_to_string(path)?;
    let mut dim = 0usize;
    let mut coords: Vec<(f64, f64)> = Vec::new();

    for line in text.lines() {
        let u = line.trim().to_uppercase();
        if u.starts_with("DIMENSION") {
            let after = line.split(':').nth(1).ok_or(TsplibError::Parse("DIMENSION"))?;
            let num_str: String = after.chars().filter(|c| c.is_ascii_digit()).collect();
            if !num_str.is_empty() {
                dim = num_str.parse().map_err(|_| TsplibError::Parse("DIMENSION"))?;
                break;
            }
        }
    }
    if dim == 0 {
        return Err(TsplibError::Parse("DIMENSION not found"));
    }

    let section = "NODE_COORD_SECTION";
    let lower = text.to_ascii_lowercase();
    let sec_pos = lower
        .find(&section.to_ascii_lowercase())
        .ok_or(TsplibError::Parse("NODE_COORD_SECTION"))?;
    coords.reserve(dim);

    for line in text[sec_pos..].lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ul = line.to_uppercase();
        if ul.starts_with("EOF") || ul.starts_with("DISPLAY_DATA_SECTION") {
            break;
        }
        let mut parts = line.split_whitespace();
        let _id = parts.next();
        let x: f64 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(TsplibError::Parse("coord x"))?;
        let y: f64 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(TsplibError::Parse("coord y"))?;
        coords.push((x, y));
        if coords.len() >= dim {
            break;
        }
    }

    if coords.len() != dim {
        return Err(TsplibError::Parse("coord count != DIMENSION"));
    }

    let mut data = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let d = if i == j {
                0.0
            } else {
                let (x1, y1) = coords[i];
                let (x2, y2) = coords[j];
                euc_2d(x1, y1, x2, y2)
            };
            data[i * dim + j] = d;
        }
    }

    Ok(DistanceMatrix::from_square(data, dim))
}
