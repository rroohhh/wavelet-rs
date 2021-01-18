use std::ops::{Add, Deref, DerefMut, Shr, Sub, Div, Mul};
use std::sync::Arc;

#[derive(Clone)]
struct QuantizationConfig<T> {
    lf: T,
    hfx: T,
    hfy: T,
    hfxy: T,
}

struct WaveletConfig<'a, T> {
    quantization_config: &'a [QuantizationConfig<T>],
}

impl<'a, T> WaveletConfig<'a, T> {
    fn new(quantization_config: &'a [QuantizationConfig<T>]) -> Self {
        Self {
            quantization_config,
        }
    }
}

struct Image<D: Deref<Target = [T]>, T, B: IntermediateType> {
    bit_depth: B,
    w: usize,
    h: usize,
    data: D,
}

impl<D: Deref<Target = [T]> + Clone, T, B: IntermediateType + Clone> Clone for Image<D, T, B> {
    fn clone(&self) -> Self {
        Self {
            bit_depth: self.bit_depth.clone(),
            w: self.w,
            h: self.h,
            data: self.data.clone()
        }
    }
}

impl<B: IntermediateType> Image<DataContainer<Vec<u16>>, u16, B> {
    fn split_from_raw(image: &rawloader::RawImage) -> Vec<(char, Self)> {
        let cfa = image.cfa.to_string();
        let data = match &image.data {
            rawloader::RawImageData::Integer(data) => data,
            _ => unimplemented!(),
        };
        let stride = (cfa.len() as f64).sqrt() as usize;
        let plane_width = image.width / stride;
        let plane_height = image.height / stride;
        let mut outputs = vec![];

        for (i, c) in cfa.chars().enumerate() {
            let mut output = vec![0; data.len() / cfa.len()];
            let offx = i % stride;
            let offy = i / stride;

            for y in 0..plane_height {
                for x in 0..plane_width {
                    output[y * plane_width + x]
                        = data[(y * stride + offy) * image.width + x * stride + offx];
                }
            }

            outputs.push((
                c, Self {
                    w: plane_width,
                    h: plane_height,
                    data: DataContainer::new(output),
                    bit_depth: B::new(),
            }))
        }

        outputs
    }
}

impl<D: Deref<Target = [T]>, T, B: IntermediateType> Image<D, T, B> {
    fn new(w: usize, h: usize, data: D, bit_depth: B) -> Self {
        Self {
            w,
            h,
            data,
            bit_depth,
        }
    }
}

struct WaveletTransformer<'a, T> {
    config: WaveletConfig<'a, T>,
}

impl<'a, Q: Copy> WaveletTransformer<'a, Q> {
    fn new(config: WaveletConfig<'a, Q>) -> Self {
        Self { config }
    }

    fn transform_with_two_buffers<T, D, B, O, G>(
        &self,
        image: Image<D, T, B>,
        output: &mut O,
        temporary: &mut O,
    )
    where
        D: Deref<Target = [T]>,
        T: AsIntermediateFor<B, T> + Copy,
        B: IntermediateType<T = Q>,
        O: DerefMut<Target = [G]>,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
            + Shr<B::T, Output = B::T>
            + Div<B::T, Output = B::T>
            + Mul<B::T, Output = B::T>
            + std::fmt::Display,
    {
        macro_rules! wavelet1d {
            (@horizontal $input:ident, $output:ident, $first:ident, $second:ident, $firstlen:ident, $secondlen:ident, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$off:ident| $idx_expr:expr) => {
                for $first in 0..$firstlen {
                    let $second = 0;
                    wavelet1d!(@calculation B::ONE, B::ONE, $first, $second, $secondlen, $input, $output, $secondlen - 1, $secondlen - 2, 2, 3, $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                    for $second in (2..($secondlen - 2)).step_by(2) {
                        wavelet1d!(@calculation B::ONE, B::ONE, $first, $second, $secondlen, $input, $output, -1, -2, 2, 3, $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                    }
                    let $second = $secondlen - 2;
                    wavelet1d!(@calculation B::ONE, B::ONE, $first, $second, $secondlen, $input, $output, -1, -2, -($secondlen as isize - 2), -($secondlen as isize - 3), $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                }
            };
            (@vertical $quantization:expr, $input:ident, $output:ident, $first:ident, $second:ident, $firstlen:ident, $secondlen:ident, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$off:ident| $idx_expr:expr) => {
                for $second in (0..$secondlen).step_by(2) {
                    let (a, b, c, d) : (isize, isize, isize, isize) = if $second == 0 {
                        (($secondlen - 1) as isize, ($secondlen - 2) as isize, 2, 3)
                    } else if $second == $secondlen - 2 {
                        (-1, -2, -($secondlen as isize - 2), -($secondlen as isize - 3))
                    } else {
                        (-1, -2, 2, 3)
                    };

                    for $first in 0..($firstlen / 2) {
                        wavelet1d!(@calculation $quantization.lf, $quantization.hfy, $first, $second, $secondlen, $input, $output, a, b, c, d, $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                    }

                    for $first in ($firstlen / 2)..$firstlen {
                        wavelet1d!(@calculation $quantization.hfx, $quantization.hfxy, $first, $second, $secondlen, $input, $output, a, b, c, d, $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                    }
                }
            };
            (@calculation $lf_quant:expr, $hf_quant:expr, $first:ident, $second:ident, $secondlen:ident, $input:ident, $output:ident, $a:expr, $b:expr, $c:expr, $d:expr, $vm:ident, $map_expr:expr, $vu:ident, $unmap_expr:expr, $off:ident, $idx_expr:expr) => {
                let pxm1 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $a, $idx_expr)], $map_expr);
                let pxm2 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $b, $idx_expr)], $map_expr);

                let pxp0 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 0, $idx_expr)], $map_expr);
                let pxp1 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 1, $idx_expr)], $map_expr);

                let pxp2 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $c, $idx_expr)], $map_expr);
                let pxp3 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $d, $idx_expr)], $map_expr);
                let lf_rounding = ($lf_quant >> B::ONE);
                let hf_rounding = ($hf_quant >> B::ONE);
                // println!("{}", lf_rounding);
                // println!("{}", hf_rounding);
                // eprintln!("{}, {}", wavelet1d!(@off $off, -($second as isize / 2), $idx_expr), $lf_quant);
                // eprintln!("{}, {}", wavelet1d!(@off $off, -($second as isize / 2) + $secondlen as isize / 2, $idx_expr), $hf_quant);
                let lf_value = pxp0 + pxp1;
                let lf_value = (lf_value + lf_rounding * <B as IntermediateType>::signum(lf_value)) / $lf_quant;
                let hf_value = pxp0 - pxp1 + ((pxp2 + pxp3 - pxm1 - pxm2 + B::FOUR) >> B::THREE);
                let hf_value = (hf_value + hf_rounding * <B as IntermediateType>::signum(hf_value)) / $hf_quant;

                $output[wavelet1d!(@off $off, -($second as isize / 2), $idx_expr)] =
                    wavelet1d!(@unmap $vu, lf_value, $unmap_expr);
                $output[wavelet1d!(@off $off, -($second as isize / 2) + $secondlen as isize / 2, $idx_expr)] =
                    wavelet1d!(@unmap $vu, hf_value, $unmap_expr);
            };
            (@map $v:ident, $value:expr, $map_expr:expr) => {
                {
                    let $v = $value;
                    $map_expr
                }
            };
            (@unmap $v:ident, $value:expr, $unmap_expr:expr) => {
                {
                    let $v = $value;
                    $unmap_expr
                }
            };
            (@off $off:ident, $value:expr, $idx_expr:expr) => {
                {
                    let $off = $value;
                    $idx_expr
                }
            }
        }

        assert!(output.len() >= image.data.len());
        assert!(temporary.len() >= image.data.len());

        let w = image.w;
        let h = image.h;
        let input = image.data;

        let mut rectw = w;
        let mut recth = h;

        let level = self.config.quantization_config.len();

        for i in 0..level {
            let quantization = &self.config.quantization_config[i];

            eprintln!("forward horizontal {}", i);
            if i == 0 {
                wavelet1d!(@horizontal
                    input,
                    temporary,
                    y,
                    x,
                    recth,
                    rectw,
                    |v| v.as_intermediate(),
                    |v| <G as CanBeOutputFor<B, G>>::map(v),
                    |off| y * w + (x as isize + off as isize) as usize
                );
            } else {
                wavelet1d!(@horizontal
                    output,
                    temporary,
                    y,
                    x,
                    recth,
                    rectw,
                    |v| <G as CanBeOutputFor<B, G>>::unmap(v),
                    |v| <G as CanBeOutputFor<B, G>>::map(v),
                    |off| y * w + (x as isize + off as isize) as usize
                );
            }

            eprintln!("forward vertical {}", i);
            wavelet1d!(@vertical
                quantization,
                temporary,
                output,
                x,
                y,
                rectw,
                recth,
                |v| <G as CanBeOutputFor<B, G>>::unmap(v),
                |v| <G as CanBeOutputFor<B, G>>::map(v),
                |off| ((y as isize + off as isize) as usize) * w + x
            );

            rectw /= 2;
            recth /= 2;
        }
    }

    fn transform<T, D, B, O, G>(&self, image: Image<D, T, B>, output: &mut O)
    where
        D: Deref<Target = [T]>,
        T: AsIntermediateFor<B, T> + Copy,
        B: IntermediateType<T = Q>,
        O: DerefMut<Target = [G]> + Clone,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
            + Div<B::T, Output = B::T>
            + Shr<B::T, Output = B::T>
            + Mul<B::T, Output = B::T>
            + std::fmt::Display,
    {
        let mut temp = output.clone();
        self.transform_with_two_buffers(image, output, &mut temp)
    }

    fn psnr<D, T, B, O>(&self, reference: Image<D, T, B>, decoded: &O) -> f64
        where
        D: Deref<Target = [T]>,
        T: AsIntermediateFor<B, T> + Copy,
        B: IntermediateType,
        O: DerefMut<Target = [B::T]>,
        B::T: Sub<B::T, Output = B::T>
            + Mul<B::T, Output = B::T>
            + std::fmt::Display,
    {
        let mut sum = 0.0;

        let ref_data = reference.data;

        for (ref_value, decoded_value) in (&ref_data).iter().zip((&decoded).iter()) {
            let diff = <B as IntermediateType>::as_i64(ref_value.as_intermediate() - *decoded_value) as f64;
            if diff != 0.0 {
                // println!("{}", diff);
            }
            sum += diff * diff;
        }

        let max_value = ((1 << B::BITDEPTH) - 1) as f64;

        10.0 * (max_value * max_value / (sum / ref_data.len() as f64)).log10()
    }

    fn rle_iter<'b, B, D, O>(encoded: &'b O, allowed_rle_words: &'b D) -> RleIter<'b, B::T>
        where
        O: Deref<Target = [B::T]>,
        D: Deref<Target = [usize]>,
        B: IntermediateType<T = Q>,
        B::T: std::cmp::PartialEq
    {
        RleIter {
            pos: 0,
            data: encoded,
            found_zeros: 0,
            rle_index: 0,
            allowed_rle_words: allowed_rle_words,
            is_nonzero: &|a| a != B::NULL
        }
    }

    fn reverse_transform_with_two_buffers<B, D, O, G>(
        &self,
        w: usize,
        h: usize,
        input: &mut D,
        output: &mut O,
    )
    where
        D: DerefMut<Target = [G]>,
        O: DerefMut<Target = [B::T]>,
        B: IntermediateType<T = Q>,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
            + Mul<B::T, Output = B::T>
            + Shr<B::T, Output = B::T>
            + std::fmt::Display,
    {
        let size = w * h;

        assert!(input.len() >= size);
        assert!(output.len() >= size);

        for y in 0..h {
            for x in 0..w {
                output[y * w + x] = <G as CanBeOutputFor<B, G>>::unmap(input[y * w + x]);
            }
        }
        macro_rules! iwavelet1d {
            (@horizontal $input:ident, $output:ident, $first:ident, $second:ident, $firstlen:expr, $secondlen:expr, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$lfoff:ident| $lf_idx_expr:expr, |$hfoff:ident| $hf_idx_expr:expr) => {
                for $first in 0..$firstlen {
                    let $second = 0;
                    iwavelet1d!(@calculate B::ONE, B::ONE, $input, $output, $second, $secondlen, $secondlen as isize - 1isize, 1, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);

                    for $second in 1..($secondlen - 1) {
                        iwavelet1d!(@calculate B::ONE, B::ONE, $input, $output, $second, $secondlen, -1isize, 1, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                    }

                    let $second = $secondlen - 1;
                    iwavelet1d!(@calculate B::ONE, B::ONE, $input, $output, $second, $secondlen, -1isize, -($secondlen as isize - 1), $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                }
            };
            (@vertical $quantization:expr, $input:ident, $output:ident, $first:ident, $second:ident, $firstlen:expr, $secondlen:expr, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$lfoff:ident| $lf_idx_expr:expr, |$hfoff:ident| $hf_idx_expr:expr) => {
                for $second in 0..$secondlen {
                    let (a, b) = if $second == 0 {
                        ($secondlen as isize - 1isize, 1)
                    } else if ($second == $secondlen - 1) {
                        (-1isize, -($secondlen as isize - 1))
                    } else {
                        (-1isize, 1)
                    };

                    for $first in 0..($firstlen / 2) {
                        iwavelet1d!(@calculate $quantization.lf, $quantization.hfy, $input, $output, $second, $secondlen, a, b, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                    }

                    for $first in ($firstlen / 2)..$firstlen {
                        iwavelet1d!(@calculate $quantization.hfx, $quantization.hfxy, $input, $output, $second, $secondlen, a, b, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                    }
                }
            };
            (@calculate $lf_quant:expr, $hf_quant:expr, $input:ident, $output:ident, $second:ident, $secondlen:tt, $a:expr, $b:expr, $vm:ident, $map_expr:expr, $lfoff:ident, $lf_idx_expr:expr, $hfoff:ident, $hf_idx_expr:expr, $vu:ident, $unmap_expr:expr) => {
                let lfm1 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, $a, $lf_idx_expr)], $map_expr) * $lf_quant;
                let lfp1 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, $b, $lf_idx_expr)], $map_expr) * $lf_quant;
                let lfp0 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, 0, $lf_idx_expr)], $map_expr) * $lf_quant;

                let hf = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $hfoff, $secondlen, $hf_idx_expr)], $map_expr) * $hf_quant;

                let real1 = (((lfm1 - lfp1 + B::FOUR) >> B::THREE) + hf + lfp0) >> B::ONE;
                let real2 = (((lfp1 - lfm1 + B::FOUR) >> B::THREE) - hf + lfp0) >> B::ONE;

                $output[iwavelet1d!(@eval $hfoff, $second, $hf_idx_expr)] = iwavelet1d!(@eval $vu, real1, $unmap_expr);
                $output[iwavelet1d!(@eval $hfoff, $second + 1, $hf_idx_expr)] = iwavelet1d!(@eval $vu, real2, $unmap_expr);
            };
            (@eval $v:ident, $value:expr, $map_expr:expr) => {
                {
                    let $v = $value;
                    $map_expr
                }
            };
        }

        let level = self.config.quantization_config.len();

        let mut regw = w / (1 << level);
        let mut regh = h / (1 << level);

        for i in 0..level {
            eprintln!("inverse vertical {}", i);
            iwavelet1d!(@vertical
                &self.config.quantization_config[i],
                output,
                input,
                x,
                y,
                regw * 2,
                regh,
                |v| v,
                |v| <G as CanBeOutputFor<B, G>>::map(v),
                |off| ((y as isize + off as isize) as usize) * w + x,
                |off| (y + off) * w + x
            );
            eprintln!("inverse horizontal {}", i);
            iwavelet1d!(@horizontal
                input,
                output,
                y,
                x,
                regh * 2,
                regw,
                |v| <G as CanBeOutputFor<B, G>>::unmap(v),
                |v| v,
                |off| y * w + (x as isize + off as isize) as usize,
                |off| y * w + x + off
            );
            regw *= 2;
            regh *= 2;
        }
    }
}

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Clone)]
enum RleIterItem<T: Hash + Ord + Eq + PartialEq + Clone> {
    NonZero(T),
    Zero(usize)
}

struct RleIter<'a, T> {
    data: &'a dyn Deref<Target = [T]>,
    pos: usize,
    found_zeros: usize,
    rle_index: isize,
    // allowed_rle_words must be sorted
    allowed_rle_words: &'a dyn Deref<Target = [usize]>,
    is_nonzero: &'a dyn Fn(T) -> bool
}

impl<'a, T: Copy + Hash + Ord + Eq + PartialEq> Iterator for RleIter<'a, T> {
    type Item = RleIterItem<T>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.data.len()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        fn next_suitable_rle_word<'b, T>(s: &mut RleIter<'b, T>, num_zeros: usize) -> usize {
            loop {
                if s.rle_index >= 0 {
                    if num_zeros >= s.allowed_rle_words[s.rle_index as usize] {
                        return s.allowed_rle_words[s.rle_index as usize]
                    } else {
                        s.rle_index -= 1;
                    }
                } else {
                    return 1
                }
            }
        }

        // println!("pos {}, found_zeros {}, rle_index {}", self.pos, self.found_zeros, self.rle_index);

        if (self.pos == self.data.len()) && (self.found_zeros == 0) {
            None
        } else if self.found_zeros > 0 {
            let next_rle_word =  next_suitable_rle_word(self, self.found_zeros);
            self.found_zeros -= next_rle_word;

            Some(RleIterItem::Zero(next_rle_word))
        } else {
            let mut next_value = self.data[self.pos];

            self.found_zeros = 0;
            self.rle_index = 0;
            while !(self.is_nonzero)(next_value) {
                self.found_zeros += 1;
                if ((self.rle_index + 1) < self.allowed_rle_words.len() as isize) && (self.found_zeros > self.allowed_rle_words[self.rle_index as usize]) {
                    self.rle_index += 1;
                }

                self.pos += 1;
                if (self.pos + 1) < self.data.len() {
                    next_value = self.data[self.pos];
                } else {
                    break
                }
            }

            if self.found_zeros > 0 {
                let next_rle_word =  next_suitable_rle_word(self, self.found_zeros);
                self.found_zeros -= next_rle_word;

                Some(RleIterItem::Zero(next_rle_word))
            } else {
                self.pos += 1;
                Some(RleIterItem::NonZero(next_value))
            }
        }
    }
}

mod BitDepth {
    #[derive(Clone)]
    pub struct U8;
    #[derive(Clone)]
    pub struct U9;
    #[derive(Clone)]
    pub struct U12;
}

trait IntermediateType {
    type T: Copy;
    const BITDEPTH: u8;
    const NULL: Self::T;
    const ONE: Self::T;
    const FOUR: Self::T;
    const THREE: Self::T;

    fn as_i64(s: Self::T) -> i64;
    fn signum(s: Self::T) -> Self::T;

    fn new() -> Self;
}

impl IntermediateType for BitDepth::U8 {
    type T = i16;
    const BITDEPTH: u8 = 8;
    const NULL: Self::T = 0 as _;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;

    fn as_i64(s: Self::T) -> i64 { s as _ }
    fn signum(s: Self::T) -> Self::T { s.signum() }

    fn new() -> Self { Self }
}

impl IntermediateType for BitDepth::U9 {
    type T = i16;
    const BITDEPTH: u8 = 9;
    const NULL: Self::T = 0 as _;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;

    fn as_i64(s: Self::T) -> i64 { s as _ }
    fn signum(s: Self::T) -> Self::T { s.signum() }

    fn new() -> Self { Self }
}

impl IntermediateType for BitDepth::U12 {
    type T = i32;
    const BITDEPTH: u8 = 12;
    const NULL: Self::T = 0 as _;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;

    fn as_i64(s: Self::T) -> i64 { s as _ }
    fn signum(s: Self::T) -> Self::T { s.signum() }

    fn new() -> Self { Self }
}

trait CanBeOutputFor<B: IntermediateType, O> {
    fn map(i: B::T) -> O;
    fn unmap(self) -> B::T;
}

macro_rules! impl_can_be_output_for {
    ($b:ty, $t:ty) => {
        impl_can_be_output_for!($b, $t, i => i, self => self);
    };
    ($b:ty, $t:ty, $i:ident => $f:expr, $self:ident => $g:expr) => {
        impl CanBeOutputFor<$b, $t> for $t {
            fn map($i: <$b as IntermediateType>::T) -> $t {
                $f as _
            }

            fn unmap($self) -> <$b as IntermediateType>::T {
                $g as _
            }
        }
    }
}

impl_can_be_output_for!(BitDepth::U8, i16);
impl_can_be_output_for!(BitDepth::U8, u16, i => i as i32 - i16::MIN as i32, self => self as i32 + i16::MIN as i32);
impl_can_be_output_for!(BitDepth::U8, u32, i => i as i32 - i16::MIN as i32, self => self as i32 + i16::MIN as i32);
impl_can_be_output_for!(BitDepth::U8, i32);
impl_can_be_output_for!(BitDepth::U12, i32);
impl_can_be_output_for!(BitDepth::U12, u32, i => i as i32 - i32::MIN, self => self as i64 + i32::MIN as i64);

trait AsIntermediateFor<B: IntermediateType, O> {
    fn as_intermediate(self) -> B::T;
}

macro_rules! impl_as_intermediate_for {
    ($b:ty, $t:ty) => {
        impl AsIntermediateFor<$b, $t> for $t {
            fn as_intermediate(self) -> <$b as IntermediateType>::T {
                self as _
            }
        }
    };
}

impl_as_intermediate_for!(BitDepth::U8, u8);
impl_as_intermediate_for!(BitDepth::U8, u16);
impl_as_intermediate_for!(BitDepth::U8, u32);

impl_as_intermediate_for!(BitDepth::U12, u16);
impl_as_intermediate_for!(BitDepth::U12, u32);

use std::fs::File;
use std::io::BufWriter;

#[derive(Clone)]
struct DataContainer<D> {
    data: Arc<D>
}

impl<D> DataContainer<D> {
    fn new(data: D) -> Self {
        Self {
            data: Arc::new(data)
        }
    }
}

impl<D> Deref for DataContainer<D>
    where D: Deref
 {
    type Target = D::Target;

    fn deref(&self) -> &Self::Target {
        &*self.data
    }
}

mod huffman {
    use std::ops::{Deref, Add};

    pub fn generate_huffman_table<T: Deref<Target = [D]>, D: Ord + Clone + Add<D, Output = D>>(frequencies: T) -> Vec<String> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Eq, PartialEq, Clone)]
        struct Node<W> {
            weight: W,
            idx: usize,
            left_right: Option<(usize, usize)>,
        }

        impl<W: Clone + Add<W, Output = W> + PartialOrd> Node<W> {
            fn new(idx: usize, weight: &W) -> Self {
                Self {
                    weight: weight.clone(),
                    idx,
                    left_right: None,
                }
            }

            fn combine(idx: usize, a: &Self, b: &Self) -> Self {
                let (right, left) = if a.weight >= b.weight {
                    (a, b)
                } else {
                    (b, a)
                };

                let left = left.idx;
                let right = right.idx;

                Self {
                    idx,
                    weight: a.weight.clone() + b.weight.clone(),
                    left_right: Some((left, right)),
                }
            }
        }

        impl<W: Ord> Ord for Node<W> {
            fn cmp(&self, other: &Self) -> Ordering {
                // min heap
                other.weight.cmp(&self.weight)
            }
        }

        impl<W: Ord> PartialOrd for Node<W> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut nodes = Vec::with_capacity(frequencies.len() * 2);

        for (i, freq) in frequencies.iter().enumerate() {
            nodes.push(Node::new(i, freq));
        }

        let mut heap = BinaryHeap::new();

        for node in &nodes {
            heap.push(node.clone());
        }

        while heap.len() > 1 {
            let new = Node::combine(nodes.len(), &heap.pop().unwrap(), &heap.pop().unwrap());
            nodes.push(new);
            heap.push(nodes.last().unwrap().clone());
        }

        let root = heap.pop().unwrap();


        fn find_huffman_codes<W>(prefix: String, nodes: &Vec<Node<W>>, node: usize, storage: &mut Vec<String>) {
            let node = &nodes[node];
            match node.left_right {
                None => {
                    storage[node.idx] = prefix;
                },
                Some((left, right)) => {
                    find_huffman_codes(prefix.clone() + "0", nodes, left, storage);
                    find_huffman_codes(prefix + "1", nodes, right, storage);
                }
            }
        }

        let mut huffman_codes = vec!["".to_owned(); frequencies.len()];
        find_huffman_codes("".to_owned(), &nodes, root.idx, &mut huffman_codes);

        return huffman_codes;
    }
}

use std::hash::Hash;
use std::collections::HashMap;

fn calculate_huffman_size<T: Ord + Hash + Clone>(data: &Vec<T>) -> usize {

    let mut symbol_counts = HashMap::new();

    for symbol in data {
        symbol_counts.insert(symbol, match symbol_counts.get(symbol) {
            Some(count) => count + 1,
            None => 0
        });
    }

    let mut symbols = Vec::new();
    let mut counts = Vec::new();

    for (symbol, count) in symbol_counts.into_iter() {
        symbols.push(symbol);
        counts.push(count);
    }

    let huffman_symbols = huffman::generate_huffman_table(counts);
    let mut huffman_table = HashMap::<T, usize>::new();

    for (symbol, huffman_symbol) in symbols.into_iter().zip(huffman_symbols.into_iter()) {
        huffman_table.insert(symbol.clone(), huffman_symbol.len());
    }

    let mut size = 0;
    for symbol in data {
        size += huffman_table[symbol];
    }

    size / 8
}

fn main() {
    let frequencies = vec![2, 6, 7, 10, 10, 11];
    println!("{:?}", huffman::generate_huffman_table(frequencies));

    let image = rawloader::decode_file("axiomlabs.dng").unwrap();

    let images = Image::<_, _, BitDepth::U12>::split_from_raw(&image);

    let mut out = vec![0i32; image.width * image.height];
    let mut tmp = vec![0i32; image.width * image.height];
    let mut decoded = vec![0i32; image.width * image.height];

    let wavelet_config = vec![QuantizationConfig { lf: 1, hfx: 32, hfy: 32, hfxy: 32 }; 3];
    let wavelet_transformer = WaveletTransformer::new(WaveletConfig::new(&wavelet_config));

    let N = 1;
    let now = std::time::Instant::now();
    for (i, (c, image)) in images.iter().enumerate() {
        println!("processing {}", c);
        wavelet_transformer.transform_with_two_buffers(image.clone(), &mut out, &mut tmp);

        let mut rle_symbols = Vec::with_capacity(out.len());
        rle_symbols.extend(WaveletTransformer::rle_iter::<BitDepth::U12, _, _>(&out, &(1..4096).collect::<Vec<_>>()));
        // let rle_symbols = .collect::<Vec<_>>();
        println!("image pixels: {}, rle_elements: {}", image.data.len(), rle_symbols.len());
        let huffman_length = calculate_huffman_size(&rle_symbols);
        println!("original size: {}, huffman size: {}", image.w * image.h * 3 / 4, huffman_length);
        // write32(out.clone(), &format!("tree-{}-{}.tiff", i, c), image.w as u32, image.h as u32);

        wavelet_transformer.reverse_transform_with_two_buffers::<BitDepth::U12, _, _, _>(
            image.w as usize,
            image.h as usize,
            &mut out,
            &mut decoded,
        );

        // println!("psnr: {}", wavelet_transformer.psnr(image.clone(), &decoded));

        // for i in 0..image.data.len() {
        //     if image.data[i] as i32 != decoded[i] {
        //         println!("fuckup");
        //         break;
        //     }
        // }

        // write16(decoded.clone(), &format!("decoded-{}-{}.tiff", i, c), image.w as u32, image.h as u32);
    }
    println!("{}", now.elapsed().as_secs_f64() / N as f64);
}

fn write16(d: Vec<i32>, name: &str, width: u32, height: u32) {
    let mut encoder =
        tiff::encoder::TiffEncoder::new(BufWriter::new(File::create(name).unwrap())).unwrap();
    let image = encoder
        .new_image::<tiff::encoder::colortype::Gray16>(width, height)
        .unwrap();
    let mut out_data = vec![0; d.len()];

    for i in 0..d.len() {
        out_data[i] = (d[i] as i64 - i32::MIN as i64) as u16;
    }

    image.write_data(&out_data).unwrap();
}

fn write32(d: Vec<i32>, name: &str, width: u32, height: u32) {
    let mut encoder =
        tiff::encoder::TiffEncoder::new(BufWriter::new(File::create(name).unwrap())).unwrap();
    let image = encoder
        .new_image::<tiff::encoder::colortype::Gray32>(width, height)
        .unwrap();
    let mut out_data = vec![0; d.len()];

    for i in 0..d.len() {
        out_data[i] = (d[i] as i64 - i32::MIN as i64) as u32;
    }

    image.write_data(&out_data).unwrap();
}
