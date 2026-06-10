//! `rustcox info <TYPE>` — print group metadata.

use anyhow::Context;
use rustcox_core::group::CoxeterGroup;

pub fn run(type_str: &str) -> anyhow::Result<()> {
    let group =
        CoxeterGroup::from_type(type_str).with_context(|| format!("invalid type '{type_str}'"))?;

    println!("type:   {type_str}");
    println!("rank:   {}", group.rank);
    println!("order:  {}", group.order);
    println!("N:      {}", group.n_pos);

    let degrees: Vec<String> = group.degrees.iter().map(|d| d.to_string()).collect();
    println!("degrees: [{}]", degrees.join(", "));

    println!("coxeter matrix:");
    for row in &group.coxmat {
        let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        println!("  [{}]", row_str.join(", "));
    }

    Ok(())
}
