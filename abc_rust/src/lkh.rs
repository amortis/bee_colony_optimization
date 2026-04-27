//! Wrapper around the LKH-3 / LKH-2 solver binary (same role as `LKH/run_lkh.py`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
pub enum LkhError {
    BinaryNotFound(PathBuf),
    Io(std::io::Error),
    Timeout,
    Parse,
    NonZeroExit(i32),
}

impl std::fmt::Display for LkhError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LkhError::BinaryNotFound(p) => write!(f, "LKH binary not found: {}", p.display()),
            LkhError::Io(e) => write!(f, "{e}"),
            LkhError::Timeout => write!(f, "LKH subprocess timeout"),
            LkhError::Parse => write!(f, "could not parse LKH cost from output"),
            LkhError::NonZeroExit(c) => write!(f, "LKH exited with status {c}"),
        }
    }
}

impl std::error::Error for LkhError {}

impl From<std::io::Error> for LkhError {
    fn from(e: std::io::Error) -> Self {
        LkhError::Io(e)
    }
}

#[derive(Debug, Clone)]
pub struct LkhResult {
    pub cost: f64,
    pub time_sec: f64,
    pub stdout_stderr: String,
}

pub struct LkhRunner {
    pub lkh_binary: PathBuf,
}

impl LkhRunner {
    /// `lkh_binary`: path to the `LKH` executable (e.g. `.../LKH/LKH-2.0.7/LKH`).
    pub fn new(lkh_binary: impl Into<PathBuf>) -> Self {
        Self {
            lkh_binary: lkh_binary.into(),
        }
    }

    /// Default relative to the `bee_colony_optimization` repo root: `LKH/LKH-2.0.7/LKH`.
    pub fn from_repo_root(repo_root: &Path) -> Self {
        Self::new(repo_root.join("LKH").join("LKH-2.0.7").join("LKH"))
    }

    pub fn binary_exists(&self) -> bool {
        self.lkh_binary.exists()
    }

    /// Run LKH with a generated `.par` file (same fields as `run_lkh.py`).
    pub fn run(
        &self,
        tsp_file: &Path,
        work_dir: &Path,
        max_trials: u64,
        time_limit_sec: u64,
        seed: u64,
    ) -> Result<LkhResult, LkhError> {
        if !self.binary_exists() {
            return Err(LkhError::BinaryNotFound(self.lkh_binary.clone()));
        }

        fs::create_dir_all(work_dir)?;

        let tsp_abs = tsp_file.canonicalize()?;
        let stem = tsp_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("problem");
        let tour_file = work_dir.join(format!("{stem}.tour"));
        let par_file = work_dir.join(format!("{stem}.par"));

        let par_content = format!(
            "PROBLEM_FILE = {}\nRUNS = 1\nSEED = {}\nMAX_TRIALS = {}\nTIME_LIMIT = {}\nOUTPUT_TOUR_FILE = {}\n",
            tsp_abs.display(),
            seed,
            max_trials,
            time_limit_sec,
            tour_file.display()
        );
        fs::write(&par_file, par_content)?;

        let t0 = Instant::now();
        let output = Command::new(&self.lkh_binary)
            .arg(&par_file)
            .output()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let text = String::from_utf8_lossy(&output.stdout).to_string()
            + &String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return Err(LkhError::NonZeroExit(
                output.status.code().unwrap_or(-1),
            ));
        }

        let cost = parse_cost(&text).ok_or(LkhError::Parse)?;

        Ok(LkhResult {
            cost,
            time_sec: elapsed,
            stdout_stderr: text,
        })
    }
}

fn parse_cost(output: &str) -> Option<f64> {
    for line in output.lines() {
        let line = line.trim();
        if line.to_uppercase().contains("COST") && line.contains('=') {
            let rest = line.split('=').nth(1)?.trim();
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            return num.parse().ok();
        }
    }
    None
}
