use ndarray::AssignElem;

#[derive(Clone)]
pub enum CounterEst<T>
where
    T: std::ops::Add<T, Output = T>,
{
    Accurate(T),
    Estimate(T),
    None,
}

impl<T> std::ops::AddAssign<CounterEst<T>> for CounterEst<T>
where
    T: std::ops::Add<T, Output = T> + Clone,
{
    fn add_assign(&mut self, rhs: CounterEst<T>) {
        let res = match rhs {
            Self::None => match self {
                Self::None => Self::None,
                Self::Accurate(val) => Self::Estimate(val.clone()),
                Self::Estimate(val) => Self::Estimate(val.clone()),
            },
            Self::Accurate(rhs_val) => match self {
                Self::None => Self::Estimate(rhs_val.clone()),
                Self::Accurate(val) => Self::Accurate(val.clone() + rhs_val.clone()),
                Self::Estimate(val) => Self::Estimate(val.clone() + rhs_val.clone()),
            },
            Self::Estimate(rhs_val) => match self {
                Self::None => Self::Estimate(rhs_val.clone()),
                Self::Accurate(val) => Self::Estimate(val.clone() + rhs_val.clone()),
                Self::Estimate(val) => Self::Estimate(val.clone() + rhs_val.clone()),
            },
        };
        self.assign_elem(res);
    }
}

impl<T> std::ops::Add<CounterEst<T>> for CounterEst<T>
where
    T: std::ops::Add<T, Output = T> + Clone,
{
    type Output = Self;

    fn add(self, rhs: CounterEst<T>) -> Self {
        let mut res = self.clone();
        res += rhs;
        res
    }
}

impl<T> std::fmt::Display for CounterEst<T>
where
    T: std::ops::Add<T, Output = T> + std::fmt::Display,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "CountNone"),
            Self::Accurate(val) => write!(f, "{} (accurate)", val),
            Self::Estimate(val) => write!(f, "{} (estimate)", val),
        }
    }
}
