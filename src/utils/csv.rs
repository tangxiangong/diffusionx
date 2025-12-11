use crate::{XResult, utils::ensure_output_dir};
use csv::Writer;
use std::path::Path;

/// Write data to a CSV file
///
/// # Arguments
///
/// * `path` - The path to the CSV file
/// * `t` - The time data
/// * `x` - The data to write
pub fn write_csv<T: ToString>(path: &str, t: &[T], x: &[T]) -> XResult<()> {
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
