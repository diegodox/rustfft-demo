use approx::assert_relative_eq;
use realfft::num_complex::Complex64;

fn main() {
    one_dim_fft();
    two_dim_fft();
}

/// In 1D Real FFT, use realfft to twice as fast as rustfft.
fn one_dim_fft() {
    // data([f64;6])
    let answer = vec![0, 1, 2, 3, 2, 1, 0]
        .into_iter()
        .map(|x| x as f64)
        .collect::<Vec<_>>();
    let mut fft_planner = realfft::RealFftPlanner::new();

    // fft
    let mut complex = {
        let fft = fft_planner.plan_fft_forward(answer.len());
        let mut f = answer.clone();
        let mut complex = fft.make_output_vec();
        fft.process(&mut f, &mut complex).unwrap();
        complex
    };

    // inv-fft
    let real = {
        let inv_fft = fft_planner.plan_fft_inverse(answer.len());
        let mut f = inv_fft.make_output_vec();
        inv_fft.process(&mut complex, &mut f).unwrap();
        let len = f.len();
        let f = f.into_iter().map(|x| x / len as f64).collect::<Vec<_>>();
        f
    };

    // test
    assert_eq!(answer.len(), real.len());
    const APPROX_EPSILON: f64 = 1e-15;
    for (ans, f) in answer.into_iter().zip(real) {
        assert_relative_eq!(ans, f, epsilon = APPROX_EPSILON)
    }
}

fn two_dim_fft() {
    // data(3x3)
    const LINE_LEN: usize = 3;
    let answer = vec![0, 1, 2, 3, 2, 1, 2, 1, 0]
        .into_iter()
        .map(|x| Complex64::new(x as f64, 0.0))
        .collect::<Vec<_>>();
    let mut fft_planner = rustfft::FftPlanner::new();

    // fft
    let mut complex = {
        let fft = fft_planner.plan_fft_forward(LINE_LEN);
        let mut f = answer.clone();

        // fft x dir
        for input in f.chunks_exact_mut(LINE_LEN) {
            fft.process(input);
        }
        // transpose fft output
        let mut t = {
            let mut t = f.clone();
            transpose::transpose(&f, &mut t, LINE_LEN, answer.len() / LINE_LEN);
            t
        };
        // fft y dir
        // reuse o as output
        for input in t.chunks_exact_mut(LINE_LEN) {
            fft.process(input);
        }
        t
    };

    // inv-fft
    let real = {
        let inv_fft = fft_planner.plan_fft_inverse(LINE_LEN);
        // inv-fft y dir
        for input in complex.chunks_exact_mut(LINE_LEN) {
            inv_fft.process(input);
        }
        // transpose inv-fft output
        let mut t = {
            let mut t = complex.clone();
            transpose::transpose(&complex, &mut t, LINE_LEN, answer.len() / LINE_LEN);
            t
        };
        // inv-fft x dir
        // reuse o as output
        for input in t.chunks_exact_mut(LINE_LEN) {
            inv_fft.process(input);
        }
        t
    };

    // test
    assert_eq!(answer.len(), real.len());
    const APPROX_EPSILON: f64 = 1e-15;
    const AMP: f64 = LINE_LEN.pow(2) as f64;
    for (ans, f) in answer.into_iter().zip(real) {
        assert_relative_eq!(ans.re, f.re / AMP, epsilon = APPROX_EPSILON);
        assert_relative_eq!(ans.im, f.im / AMP, epsilon = APPROX_EPSILON);
    }
}
