#![feature(portable_simd)]

use std::simd::*;

#[derive(Debug)]
pub enum Error {
    TryFrom,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimdU256(u32x8);

impl From<u128> for SimdU256 {
    fn from(val: u128) -> Self {
        let ary = [
            0, 0, 0, 0,
            (val << 0 >> 96) as u32,
            (val << 32 >> 96) as u32,
            (val << 64 >> 96) as u32,
            (val << 96 >> 96) as u32,
        ];
        Self(u32x8::from_array(ary))
    }
}

impl From<[u32; 8]> for SimdU256 {
    fn from(val: [u32; 8]) -> Self {
        Self(u32x8::from_array(val))
    }
}

impl From<SimdU256> for u32x8 {
    fn from(val: SimdU256) -> Self {
        val.0
    }
}

impl From<u32x8> for SimdU256 {
    fn from(val: u32x8) -> Self {
        Self(val)
    }
}

impl From<SimdU256> for [u32; 8] {
    fn from(val: SimdU256) -> Self {
        u32x8::to_array(val.0)
    }
}

impl std::convert::TryFrom<SimdU256> for u128 {
    type Error = Error;
    fn try_from(value: SimdU256) -> Result<Self, Self::Error> {
        let ary : [u32; 8] = value.into();
        if (ary[0] | ary[1] | ary[2] | ary[3]) != 0 {
            Err(Error::TryFrom)
        } else {
            Ok(
                (ary[4] as u128) << 96 |
                (ary[5] as u128) << 64 |
                (ary[6] as u128) << 32 |
                (ary[7] as u128) << 0
            )
        }
    }
}

impl std::ops::Mul<SimdU256> for SimdU256 {
    type Output = SimdU256;
    fn mul(self, rhs: SimdU256) -> Self::Output {
        let a = self.0.cast::<u64>();
        let b = rhs.0;
        let sh = u64x8::splat(32);

        let t = u64x8::splat(0);
        let t = a * simd_swizzle!(b, [0, 0, 0, 0, 0, 0, 0, 0]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [1, 1, 1, 1, 1, 1, 1, 1]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [2, 2, 2, 2, 2, 2, 2, 2]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [3, 3, 3, 3, 3, 3, 3, 3]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [4, 4, 4, 4, 4, 4, 4, 4]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [5, 5, 5, 5, 5, 5, 5, 5]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [6, 6, 6, 6, 6, 6, 6, 6]).cast::<u64>() + shl64(t);
        let t = (t << sh >> sh) + shl64(t >> sh);
        let t = a * simd_swizzle!(b, [7, 7, 7, 7, 7, 7, 7, 7]).cast::<u64>() + shl64(t);
        Self(carry_prop(t))
    }
}

impl std::ops::Mul<u32> for SimdU256 {
    type Output = SimdU256;
    fn mul(self, rhs: u32) -> Self::Output {
        let a = self.0.cast::<u64>();
        let sh = u64x8::splat(32);

        let t = a * u64x8::splat(rhs as u64);
        let t = (t << sh >> sh) + shl64(t >> sh);
        Self(carry_prop(t))
    }
}

impl std::ops::Add<SimdU256> for SimdU256 {
    type Output = SimdU256;
    fn add(self, rhs: SimdU256) -> Self::Output {
        let a = self.0.cast::<u64>();
        let b = rhs.0;
        Self(carry_prop(a.cast::<u64>() + b.cast::<u64>()))
    }
}

impl std::ops::Sub<SimdU256> for SimdU256 {
    type Output = SimdU256;
    fn sub(self, rhs: SimdU256) -> Self::Output {
        // Twos complement subtract (a + !b + 1)
        let a = self.0.cast::<u64>();
        let b = !rhs.0;
        Self(carry_prop(a + b.cast::<u64>() + u64x8::from_array([0, 0, 0, 0, 0, 0, 0, 1])))
    }
}

impl std::ops::Div<SimdU256> for SimdU256 {
    type Output = SimdU256;
    fn div(self, b: SimdU256) -> Self::Output {
        let mut res = Self::from(0);

        println!("{:x}/{:x}", self, b);
        let mut a = self;

        while a >= b {
            println!("a   ={:x}", a);
            println!("b   ={:x}", b);
            let lza = a.leading_zeros();
            let lzb = b.leading_zeros();
            println!("lza={}", lza);
            println!("lzb={}", lzb);
            // 0x80000000 <= atop,btop <= 0xffffffff
            let btop = <[u32; 8]>::from(b << lzb >> (256-32))[7];
            let atop = <[u32; 8]>::from(a << lza >> (256-32))[7];
            println!("atop={:x}", atop);
            println!("btop={:x}", btop);
            let mut z = (((atop as u64) << 31) / (btop as u64)) as u32;
            let sh = if lzb < lza + 31 {
                if lza + 31 - lzb >= 32 {
                    z = 0;
                } else {
                    z >>= lza + 31 - lzb;
                }
                0
            } else {
                lzb - lza - 31
            };
            let mut test = b * z << sh;
            let mut delta = SimdU256::from(z as u128) << sh;
            println!("z   ={:x}", z);
            println!("test={:x}", test);
            println!("delt={:x}", delta);
    
            if test < a {
                // Estimate is too low. Keep adding multiples of b.
                println!("test < a");
                while test < a {
                    test = test + (b << sh);
                    delta = delta + (SimdU256::from(1) << sh);
                }
            } else {
                // Estimate is too high. Keep subtracting multiples of b.
                println!("test > a");
                while test >= a + (b << sh) {
                    test = test - (b << sh);
                    delta = delta - (SimdU256::from(1) << sh);
                }
            }
            println!("..test={:x}", test);
            println!("..delt={:x}", delta);
            res = res + delta;
            a = test - a;
            println!("..a ={:x}", a);
        }
        res
    }
}

impl std::ops::Shl<u32> for SimdU256 {
    type Output = SimdU256;
    fn shl(self, rhs: u32) -> Self::Output {
        let zero = u32x8::splat(0);
        let t = self.0;
        use Which::*;
        let t = if (rhs & 128) != 0 {
            simd_swizzle!(t, zero, [First(4), First(5), First(6), First(7), Second(0), Second(1), Second(2), Second(3)])
        } else {
            t
        };
        let t = if (rhs & 64) != 0 {
            simd_swizzle!(t, zero, [First(2), First(3), First(4), First(5), First(6), First(7), Second(0), Second(1)])
        } else {
            t
        };
        let t = if (rhs & 32) != 0 {
            simd_swizzle!(t, zero, [First(1), First(2), First(3), First(4), First(5), First(6), First(7), Second(0)])
        } else {
            t
        };
        if rhs & 31 != 0 {
            let t1 = simd_swizzle!(t, zero, [First(1), First(2), First(3), First(4), First(5), First(6), First(7), Second(0)]);
            let sh1 = u32x8::splat(rhs & 31);
            let sh2 = u32x8::splat(32 - (rhs & 31));
            Self(t << sh1 | t1 >> sh2)
        } else {
            Self(t)
        }
    }
}

impl std::ops::Shr<u32> for SimdU256 {
    type Output = SimdU256;
    fn shr(self, rhs: u32) -> Self::Output {
        let zero = u32x8::splat(0);
        let t = self.0;
        use Which::*;
        let t = if (rhs & 128) != 0 {
            simd_swizzle!(t, zero, [Second(0), Second(1), Second(2), Second(3), First(0), First(1), First(2), First(3)])
        } else {
            t
        };
        let t = if (rhs & 64) != 0 {
            simd_swizzle!(t, zero, [Second(0), Second(1), First(0), First(1), First(2), First(3), First(4), First(5)])
        } else {
            t
        };
        let t = if (rhs & 32) != 0 {
            simd_swizzle!(t, zero, [Second(0), First(0), First(1), First(2), First(3), First(4), First(5), First(6)])
        } else {
            t
        };
        if rhs & 31 != 0 {
            let t1 = simd_swizzle!(t, zero, [Second(0), First(0), First(1), First(2), First(3), First(4), First(5), First(6)]);
            let sh1 = u32x8::splat(rhs & 31);
            let sh2 = u32x8::splat(32 - (rhs & 31));
            Self(t >> sh1 | t1 << sh2)
        } else {
            Self(t)
        }
    }
}

pub fn shl64(x: u64x8) -> u64x8 {
    use Which::*;
    let zero = u64x8::splat(0);
    simd_swizzle!(x, zero, [First(1), First(2), First(3), First(4), First(5), First(6), First(7), Second(0)])
}

pub fn carry_prop(t: u64x8) -> u32x8 {
    let sh = u64x8::splat(32);
    // TODO: It must be possible to do the carry propogation in O(log(N))
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    let t = (t << sh >> sh) + shl64(t >> sh);
    t.cast::<u32>()
}

impl SimdU256 {
    pub fn to_str_radix(&self, radix:u32) -> String {
        assert!(radix == 16);
        let a = <[u32; 8]>::from(self.0);
        format!("{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }

    pub fn leading_zeros(&self) -> u32 {
        let lanes = u32x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        let seven = u32x8::from_array([7, 7, 7, 7, 7, 7, 7, 7]);
        let zero = u32x8::from_array([0, 0, 0, 0, 0, 0, 0, 0]);
        let lane = self.0.lanes_ne(zero)
            .select(lanes, seven)
            .reduce_min();
        lane*32 + self.0[lane as usize].leading_zeros()
    }

    pub fn bits(&self) -> u32 {
        256 - self.leading_zeros()
    }

    pub fn is_zero(&self) -> bool {
        let zero = u32x8::from_array([0, 0, 0, 0, 0, 0, 0, 0]);
        !self.0.lanes_ne(zero).any()
    }
}


impl std::fmt::Display for SimdU256 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.pad_integral(true, "", &self.to_str_radix(10))
    }
}

impl std::fmt::LowerHex for SimdU256 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.pad_integral(true, "0x", &self.to_str_radix(16))
    }
}

impl std::fmt::UpperHex for SimdU256 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = self.to_str_radix(16);
        s.make_ascii_uppercase();
        f.pad_integral(true, "0x", &s)
    }
}



#[test]
fn test_mul_overflow() {
    let a = u32x8::from_array([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    let b = u32x8::from_array([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    println!("{:08x?}", a);
    println!("{:08x?}", b);
    let c = SimdU256(a) * SimdU256(b);
    println!("{:08x?}", c.0);

    let a = u32x8::from_array([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    let b = u32x8::from_array([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    println!("{:08x?}", a);
    println!("{:08x?}", b);
    let c = SimdU256(a) * SimdU256(b);
    println!("{:08x?}", c.0);
}

#[test]
fn test_mul_random() {
    use num_bigint::BigUint;
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mask = BigUint::from_slice(&[0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    let extra = BigUint::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 1]);

    let to_simd = move |b: BigUint| {
        let mut r = (b & (&mask)).to_u32_digits();
        while r.len() < 8 {
            r.push(0);
        }
        r.reverse();
        SimdU256::from(<[u32; 8]>::try_from(&*r).unwrap())
    };

    for _ in 0..100000 {
        let lhs = [
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        ];
        let rhs = [
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
            rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        ];
        let mut lhsr = lhs;
        let mut rhsr = rhs;
        lhsr.reverse();
        rhsr.reverse();
        
        let lhs_uint = BigUint::from_slice(&lhsr);
        let rhs_uint = BigUint::from_slice(&rhsr);
        let lhs_simd = SimdU256::from(lhs);
        let rhs_simd = SimdU256::from(rhs);

        let prod_uint = to_simd(&lhs_uint * &rhs_uint);
        let prod_simd = lhs_simd * rhs_simd;
        assert_eq!(prod_uint, prod_simd);

        let prod_uint = to_simd(&lhs_uint * rhs[7]);
        let prod_simd = lhs_simd * rhs[7];
        assert_eq!(prod_uint, prod_simd);

        let sum_uint = to_simd(&lhs_uint + &rhs_uint);
        let sum_simd = lhs_simd + rhs_simd;
        assert_eq!(sum_uint, sum_simd);

        let diff_uint = to_simd(&extra + &lhs_uint - &rhs_uint);
        let diff_simd = lhs_simd - rhs_simd;
        assert_eq!(diff_uint, diff_simd);

        let sh = rng.gen::<u32>() & 255;
        let shl_uint = to_simd(&lhs_uint << sh);
        let shl_simd = lhs_simd << sh;
        assert_eq!(shl_uint, shl_simd);

        let shr_uint = to_simd(&lhs_uint >> sh);
        let shr_simd = lhs_simd >> sh;
        assert_eq!(shr_uint, shr_simd);

        assert_eq!((&lhs_uint >> sh).bits() as u32, shr_simd.bits());

        let div_uint = to_simd(&lhs_uint / (&rhs_uint >> 128));
        let div_simd = lhs_simd / (rhs_simd >> 128);
        assert_eq!(div_uint, div_simd);

    }
}
