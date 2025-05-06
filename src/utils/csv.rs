use crate::{XResult, utils::ensure_output_dir};
use csv::Writer;
use std::path::Path;

pub fn write_csv(path: &str, t: &[f64], x: &[f64]) -> XResult<()> {
    ensure_output_dir(Path::new(path))?;
    let mut writer = Writer::from_path(path)?;
    t.iter().zip(x.iter()).for_each(|(t, x)| {
        writer
            .write_record(&[t.to_string(), x.to_string()])
            .unwrap();
    });
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_csv() {
        let t = vec![1.0, 2.0, 3.0];
        let x = vec![4.0, 5.0, 6.0];
        let path = "tmp/test.csv";
        write_csv(path, &t, &x).unwrap();
    }
}
