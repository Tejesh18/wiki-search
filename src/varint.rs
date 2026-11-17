// Postings are stored as gaps between doc ids, so most numbers are small.
// Variable-byte encoding: 7 bits per byte, high bit set on all but the last.

pub fn write(buf: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

#[inline]
pub fn read(buf: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn round_trip() {
        let nums = [0u32, 1, 127, 128, 300, 16_384, u32::MAX];
        let mut buf = Vec::new();
        for &n in &nums {
            super::write(&mut buf, n);
        }
        let mut pos = 0;
        for &n in &nums {
            assert_eq!(super::read(&buf, &mut pos), n);
        }
        assert_eq!(pos, buf.len());
    }
}
