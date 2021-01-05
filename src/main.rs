use std::ops::{Add, Deref, DerefMut, Shr, Sub};
use std::sync::Arc;

struct QuantizationConfig {
    lf: u32,
    hfx: u32,
    hfy: u32,
    hfxy: u32,
}

struct WaveletConfig<'a> {
    quantization_config: &'a [QuantizationConfig],
}

impl<'a> WaveletConfig<'a> {
    fn new(quantization_config: &'a [QuantizationConfig]) -> Self {
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

struct WaveletTransformer<'a> {
    config: WaveletConfig<'a>,
}

impl<'a> WaveletTransformer<'a> {
    fn new(config: WaveletConfig<'a>) -> Self {
        Self { config }
    }

    fn transform_with_two_buffers<T, D, B, O, G>(
        &self,
        image: Image<D, T, B>,
        mut output: &mut O,
        mut temporary: &mut O,
    )
    where
        D: Deref<Target = [T]>,
        T: AsIntermediateFor<B, T> + Copy,
        B: IntermediateType,
        O: DerefMut<Target = [G]>,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
            + Shr<B::T, Output = B::T>
            + std::fmt::Display,
    {
        macro_rules! wavelet1d {
            ($input:ident, $output:ident, $first:ident, $second:ident, $firstlen:ident, $secondlen:ident, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$off:ident| $idx_expr:expr) => {
                for $first in 0..$firstlen {
                    let $second = 0;
                    let mut prev_lf = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $secondlen - 1, $idx_expr)], $map_expr) + wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $secondlen - 2, $idx_expr)], $map_expr);
                    let mut this_lf = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 0, $idx_expr)], $map_expr) + wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 1, $idx_expr)], $map_expr);

                    for $second in (0..($secondlen - 2)).step_by(2) {
                        wavelet1d!(@calculation $first, $second, $secondlen, $input, $output, prev_lf, this_lf, 2, 3, $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                    }
                    let $second = $secondlen - 2;
                    wavelet1d!(@calculation $first, $second, $secondlen, $input, $output, prev_lf, this_lf, -($secondlen as isize - 2), -($secondlen as isize - 3), $vm, $map_expr, $vu, $unmap_expr, $off, $idx_expr);
                }
            };
            (@calculation $first:ident, $second:ident, $secondlen:ident, $input:ident, $output:ident, $prev_lf:ident, $this_lf:ident, $a:expr, $b:expr, $vm:ident, $map_expr:expr, $vu:ident, $unmap_expr:expr, $off:ident, $idx_expr:expr) => {
                let pxp0 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 0, $idx_expr)], $map_expr);
                let pxp1 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, 1, $idx_expr)], $map_expr);

                let pxp2 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $a, $idx_expr)], $map_expr);
                let pxp3 = wavelet1d!(@map $vm, $input[wavelet1d!(@off $off, $b, $idx_expr)], $map_expr);
                let next_lf = pxp2 + pxp3;

                $output[wavelet1d!(@off $off, -($second as isize / 2), $idx_expr)] = wavelet1d!(@unmap $vu, $this_lf, $unmap_expr);
                $output[wavelet1d!(@off $off, -($second as isize / 2) + $secondlen as isize / 2, $idx_expr)] = wavelet1d!(@unmap $vu, pxp0 - pxp1 + ((next_lf - $prev_lf + B::FOUR) >> B::THREE), $unmap_expr);


                $prev_lf = $this_lf;
                $this_lf = next_lf;
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

        let level = 3;

        for i in 0..level {
            if i == 0 {
                wavelet1d!(
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
                wavelet1d!(
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

            wavelet1d!(
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
        B: IntermediateType,
        O: DerefMut<Target = [G]> + Clone,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
            + Shr<B::T, Output = B::T>
            + std::fmt::Display,
    {
        let mut temp = output.clone();
        self.transform_with_two_buffers(image, output, &mut temp)
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
        B: IntermediateType,
        G: CanBeOutputFor<B, G> + Copy + std::fmt::Display,
        B::T: Add<B::T, Output = B::T>
            + Sub<B::T, Output = B::T>
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
            ($input:ident, $output:ident, $first:ident, $second:ident, $firstlen:expr, $secondlen:expr, |$vm:ident| $map_expr:expr, |$vu:ident| $unmap_expr:expr, |$lfoff:ident| $lf_idx_expr:expr, |$hfoff:ident| $hf_idx_expr:expr) => {
                for $first in 0..$firstlen {
                    let $second = 0;
                    iwavelet1d!(@calculate $input, $output, $second, $secondlen, $secondlen as isize - 1isize, 1, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);

                    for $second in 1..($secondlen - 1) {
                        iwavelet1d!(@calculate $input, $output, $second, $secondlen, -1isize, 1, $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                    }

                    let $second = $secondlen - 1;
                    iwavelet1d!(@calculate $input, $output, $second, $secondlen, -1isize, -($secondlen as isize - 1), $vm, $map_expr, $lfoff, $lf_idx_expr, $hfoff, $hf_idx_expr, $vu, $unmap_expr);
                }
            };
            (@calculate $input:ident, $output:ident, $second:ident, $secondlen:tt, $a:expr, $b:expr, $vm:ident, $map_expr:expr, $lfoff:ident, $lf_idx_expr:expr, $hfoff:ident, $hf_idx_expr:expr, $vu:ident, $unmap_expr:expr) => {
                let lfm1 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, $a, $lf_idx_expr)], $map_expr);
                let lfp1 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, $b, $lf_idx_expr)], $map_expr);
                let lfp0 = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $lfoff, 0, $lf_idx_expr)], $map_expr);

                let hf = iwavelet1d!(@eval $vm, $input[iwavelet1d!(@eval $hfoff, $secondlen, $hf_idx_expr)], $map_expr);

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

        let level = 3;

        let mut regw = w / (1 << level);
        let mut regh = h / (1 << level);

        for i in 0..level {
            iwavelet1d!(
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
            iwavelet1d!(
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

mod BitDepth {
    pub struct U8;
    pub struct U9;
    pub struct U12;
}

trait IntermediateType {
    type T: Copy;
    const ONE: Self::T;
    const FOUR: Self::T;
    const THREE: Self::T;
}

impl IntermediateType for BitDepth::U8 {
    type T = i16;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;
}

impl IntermediateType for BitDepth::U9 {
    type T = i16;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;
}

impl IntermediateType for BitDepth::U12 {
    type T = i32;
    const ONE: Self::T = 1 as _;
    const FOUR: Self::T = 4 as _;
    const THREE: Self::T = 3 as _;
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

fn main() {
    let decoder = png::Decoder::new(File::open("/tmp/che_full.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    let mut image = vec![0; info.buffer_size()];
    reader.next_frame(&mut image).unwrap();

    let image_data = DataContainer::new(image);
    let image = Image::new(
        info.width as usize,
        info.height as usize,
        image_data.clone(),
        BitDepth::U8,
    );
    let mut out = vec![0i16; info.buffer_size()];
    let mut tmp = vec![0i16; info.buffer_size()];
    let mut decoded = vec![0i16; info.buffer_size()];

    let wavelet_config = vec![];
    let wavelet_transformer = WaveletTransformer::new(WaveletConfig::new(&wavelet_config));


    let now = std::time::Instant::now();
    for _ in 0..60 {
        let image = Image::new(
            info.width as usize,
            info.height as usize,
            image_data.clone(),
            BitDepth::U8,
        );

        wavelet_transformer.transform_with_two_buffers(image, &mut out, &mut tmp);
        wavelet_transformer.reverse_transform_with_two_buffers(
            info.width as usize,
            info.height as usize,
            &mut out,
            &mut decoded,
        );
    }
    println!("{}", now.elapsed().as_secs_f64());

    for i in 0..image_data.len() {
        if image_data[i] as i16 != decoded[i] {
            println!("fuckup");
        }
    }

    write(decoded, "test_decoded.tiff", info.width, info.height);
    write(out, "test.tiff", info.width, info.height);
}

fn write(d: Vec<i16>, name: &str, width: u32, height: u32) {
    let mut encoder =
        tiff::encoder::TiffEncoder::new(BufWriter::new(File::create(name).unwrap())).unwrap();
    let image = encoder
        .new_image::<tiff::encoder::colortype::Gray16>(width, height)
        .unwrap();
    let mut out_data = vec![0; d.len()];

    for i in 0..d.len() {
        out_data[i] = (d[i] as i32 - i16::MIN as i32) as u16;
    }

    image.write_data(&out_data).unwrap();
}
