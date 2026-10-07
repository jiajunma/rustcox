//! Run remotely (XMU HPC or GitHub Actions), never on the local editing host.
//! Usage: compact_bench E6 reference|compact|interval [--dump]
//! --dump writes a versioned full-table binary stream to stdout for sha256sum.
//! Compute time excludes validation/serialization. Peak RSS is read separately.
use std::io::{self, BufWriter, Write};
use std::time::Instant;

use rustcox_core::{
    compact::{compute, CompactKlTable, CompactOpts, IntervalLimits},
    enumerate::ElementTable,
    group::CoxeterGroup,
    kl::{klpolynomials_seq, KlOpts, KlTable},
    laurent::Laurent,
};
use serde_json::json;

enum Backend {
    Reference(KlTable),
    Compact(CompactKlTable),
}

fn peak_rss_kib() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?.split_whitespace().next()?.parse().ok()
    })
}

fn dump(
    out: &mut impl Write,
    elms: &ElementTable,
    pols: &[Laurent],
    mut id: impl FnMut(u32, u32) -> Option<u32>,
) -> io::Result<()> {
    out.write_all(b"rustcox-full-kl-stream-v1\0")?;
    out.write_all(&(elms.len() as u64).to_le_bytes())?;
    out.write_all(&(elms.rank as u64).to_le_bytes())?;
    for word in &elms.elms {
        out.write_all(&(word.len() as u32).to_le_bytes())?;
        out.write_all(word)?;
    }
    out.write_all(&(pols.len() as u64).to_le_bytes())?;
    for p in pols {
        out.write_all(&p.val().to_le_bytes())?;
        out.write_all(&(p.coeffs().len() as u32).to_le_bytes())?;
        for c in p.coeffs() { out.write_all(&c.to_le_bytes())?; }
    }
    let mut row = Vec::with_capacity(elms.len() * 4);
    for w in 0..elms.len() as u32 {
        row.clear();
        for y in 0..=w {
            row.extend_from_slice(&id(y, w).unwrap_or(u32::MAX).to_le_bytes());
        }
        out.write_all(&row)?;
    }
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 || (args.len() == 3 && args[2] != "--dump") {
        return Err("usage: compact_bench TYPE reference|compact|interval [--dump]".into());
    }
    let group = CoxeterGroup::from_type(&args[0])?;
    let start = Instant::now();
    let engine = match args[1].as_str() {
        "reference" => Backend::Reference(klpolynomials_seq(&group, &KlOpts::equal(group.rank))?),
        "compact" | "interval" => Backend::Compact(compute(&group, &CompactOpts {
            intervals: (args[1] == "interval").then(IntervalLimits::default),
            ..CompactOpts::default()
        })?),
        _ => return Err("engine must be reference, compact, or interval".into()),
    };
    let seconds = start.elapsed().as_secs_f64();
    let (n, npols, stats, storage) = match &engine {
        Backend::Reference(t) => {
            let pol_bytes: usize = t.rows.iter().map(|r| r.pol.len() * 4).sum();
            let mu_bytes: usize = t.rows.iter().map(|r| r.mu_present.as_ref().map_or(0, Vec::len)).sum();
            (t.n(), t.pols.len(), json!(null), json!({"pol_id_bytes": pol_bytes, "mu_flag_bytes": mu_bytes}))
        }
        Backend::Compact(t) => (t.elms.len(), t.pols.len(), json!(t.stats), json!(t.storage())),
    };
    let report = json!({
        "type": args[0], "engine": args[1], "threads": 1,
        "compute_seconds": seconds, "peak_rss_kib_after_compute": peak_rss_kib(),
        "elements": n, "polynomials": npols, "stats": stats, "storage": storage,
    });
    eprintln!("REPORT {report}");
    if args.len() == 3 {
        let validation = Instant::now();
        let mut out = BufWriter::with_capacity(1024 * 1024, io::stdout().lock());
        match &engine {
            Backend::Reference(t) => dump(&mut out, &t.elms, &t.pols, |y, w| {
                let i = t.rows[w as usize].pol[y as usize];
                (i != u32::MAX).then_some(i)
            })?,
            Backend::Compact(t) => dump(&mut out, &t.elms, &t.pols, |y, w| t.pol_id(y, w))?,
        }
        eprintln!("VALIDATION_SECONDS {}", validation.elapsed().as_secs_f64());
    } else {
        println!("{report}");
    }
    Ok(())
}
