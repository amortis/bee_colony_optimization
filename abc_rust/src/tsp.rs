//! Distance matrix and tour length.

#[derive(Clone, Debug)]
pub struct DistanceMatrix {
    n: usize,
    data: Vec<f64>,
}

impl DistanceMatrix {
    pub fn from_square(data: Vec<f64>, n: usize) -> Self {
        assert_eq!(data.len(), n * n);
        Self { n, data }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn tour_length(&self, tour: &[usize]) -> f64 {
        let n = tour.len();
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..n - 1 {
            s += self.get(tour[i], tour[i + 1]);
        }
        s += self.get(tour[n - 1], tour[0]);
        s
    }
}

pub fn fitness_from_distance(distance: f64) -> f64 {
    1.0 / (1e-9 + distance)
}
