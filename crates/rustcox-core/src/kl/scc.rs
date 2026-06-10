//! Tarjan's strongly-connected-components algorithm, shared by
//! [`cells`][super::cells] and [`wgraph`][crate::wgraph].

/// Tarjan's strongly-connected-components algorithm (iterative, no recursion
/// to avoid deep-stack blowups on large groups).
///
/// ## Parameters
/// - `adj`: adjacency list; `adj[v]` lists all direct successors of vertex `v`
///   as `u32` vertex indices.
/// - `n`: total number of vertices (must equal `adj.len()`).
///
/// ## Returns
/// `(comp_of, num_comp)` where `comp_of[v]` is the component id of vertex `v`
/// (ids assigned in reverse-finish order, 0-based) and `num_comp` is the total
/// number of SCCs.
pub(crate) fn tarjan_scc(adj: &[Vec<u32>], n: usize) -> (Vec<usize>, usize) {
    debug_assert_eq!(adj.len(), n, "tarjan_scc: adj.len() must equal n");
    const UNVISITED: u32 = u32::MAX;

    let mut index = vec![UNVISITED; n];
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut comp_of = vec![usize::MAX; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut next_index: u32 = 0;
    let mut num_comp = 0usize;

    // Explicit DFS stack: each frame is (vertex, next-child-cursor).
    let mut call: Vec<(u32, usize)> = Vec::new();

    for start in 0..n {
        if index[start] != UNVISITED {
            continue;
        }
        call.push((start as u32, 0));
        while let Some(&(v, ci)) = call.last() {
            let vu = v as usize;
            if ci == 0 {
                index[vu] = next_index;
                lowlink[vu] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[vu] = true;
            }
            if ci < adj[vu].len() {
                // Advance the cursor before recursing.
                call.last_mut()
                    .expect("call stack non-empty: we just matched on call.last()")
                    .1 = ci + 1;
                let w = adj[vu][ci];
                let wu = w as usize;
                if index[wu] == UNVISITED {
                    call.push((w, 0));
                } else if on_stack[wu] {
                    lowlink[vu] = lowlink[vu].min(index[wu]);
                }
            } else {
                // Done with v: if it is a root, pop an SCC.
                if lowlink[vu] == index[vu] {
                    loop {
                        let w = stack.pop().expect(
                            "Tarjan stack non-empty: SCC root was pushed before its component",
                        );
                        on_stack[w as usize] = false;
                        comp_of[w as usize] = num_comp;
                        if w == v {
                            break;
                        }
                    }
                    num_comp += 1;
                }
                call.pop();
                // Propagate lowlink to parent.
                if let Some(&(parent, _)) = call.last() {
                    let pu = parent as usize;
                    lowlink[pu] = lowlink[pu].min(lowlink[vu]);
                }
            }
        }
    }

    (comp_of, num_comp)
}
