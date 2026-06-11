//! Dump the whole-W1 `cell1` (in `wgraphtoklmat`/PerGen form) that
//! `relkl_input_from_table` builds, as JSON, so the PyCox oracle can be computed
//! on the EXACT same input the Rust `relklpols` test consumes.
//!
//! Run: `cargo run -p rustcox-core --example dump_cell1 > /tmp/cell1.json`.
//!
//! Output: a JSON object keyed by `<TYPE>_<J>` with fields:
//! - `type`: `[series, rank]`
//! - `J`: the kept generators (W-indices)
//! - `elms`: W1-LOCAL reduced words (the cell vertices)
//! - `mpols`: per-generator mu pools (each `[[val,c0,…], …]`)
//! - `klmat`: lower-triangular; `klmat[j][i]` is `null` (`'f'`) or the
//!   per-generator mu index vector `[i0, i1, …]`.

use rustcox_core::{
    cellgraph::{relkl_input_from_table, MuPools},
    group::CoxeterGroup,
    kl::{klpolynomials_seq, KlOpts},
    laurent::Laurent,
    parabolic::Parabolic,
};

fn laurent_json(p: &Laurent) -> String {
    if p.is_zero() {
        return "[0]".to_string();
    }
    let mut parts = vec![p.val().to_string()];
    for e in p.val()..=p.degree().unwrap() {
        parts.push(p.coeff(e).to_string());
    }
    format!("[{}]", parts.join(","))
}

fn main() {
    let cases = [
        ("A", 3usize, vec![0u8, 1]),
        ("B", 3, vec![0, 1]),
        ("A", 2, vec![0]),
        ("H", 3, vec![0, 1]),
        ("A", 4, vec![0, 1, 2]),
    ];

    let mut entries: Vec<String> = Vec::new();
    for (ty, rank, jvec) in cases {
        let w = CoxeterGroup::from_type(&format!("{ty}{rank}")).unwrap();
        let w1 = Parabolic::new(&w, &jvec).unwrap();
        let opts = KlOpts::equal(w1.group.rank);
        let t1 = klpolynomials_seq(&w1.group, &opts).unwrap();
        let all: Vec<u32> = (0..t1.n() as u32).collect();
        let cell1 = relkl_input_from_table(&w1.group, &t1, &all);

        let key = format!(
            "{ty}{rank}_{}",
            jvec.iter().map(|g| g.to_string()).collect::<String>()
        );

        let elms_json: Vec<String> = cell1
            .elms
            .iter()
            .map(|w| {
                format!(
                    "[{}]",
                    w.iter()
                        .map(|g| g.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();

        let MuPools::PerGen(pools) = &cell1.mpols else {
            unreachable!()
        };
        let pools_json: Vec<String> = pools
            .iter()
            .map(|pool| {
                let ps: Vec<String> = pool.iter().map(laurent_json).collect();
                format!("[{}]", ps.join(","))
            })
            .collect();

        let klmat_json: Vec<String> = cell1
            .klmat
            .iter()
            .map(|row| {
                let cells: Vec<String> = row
                    .iter()
                    .map(|slot| match slot {
                        None => "null".to_string(),
                        Some(sd) => format!(
                            "[{}]",
                            sd.mu
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                    })
                    .collect();
                format!("[{}]", cells.join(","))
            })
            .collect();

        entries.push(format!(
            "\"{key}\":{{\"type\":[\"{ty}\",{rank}],\"J\":[{}],\"elms\":[{}],\"mpols\":[{}],\"klmat\":[{}]}}",
            jvec.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(","),
            elms_json.join(","),
            pools_json.join(","),
            klmat_json.join(","),
        ));
    }
    println!("{{{}}}", entries.join(","));
}
