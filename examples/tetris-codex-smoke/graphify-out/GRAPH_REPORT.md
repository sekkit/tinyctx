# Graph Report - tetris-codex-smoke  (2026-05-22)

## Corpus Check
- 3 files · ~1,577 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 43 nodes · 66 edges · 7 communities (5 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7e85c96e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]

## God Nodes (most connected - your core abstractions)
1. `TetrisGame` - 22 edges
2. `Tetris Smoke` - 5 edges
3. `cloneMatrix()` - 2 edges
4. `createBoard()` - 2 edges
5. `rotateMatrix()` - 2 edges
6. `Test` - 2 edges
7. `TYPES` - 1 edges
8. `SHAPES` - 1 edges
9. `COLORS` - 1 edges
10. `POINTS` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (7 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.29
Nodes (5): COLORS, POINTS, rotateMatrix(), SHAPES, TYPES

### Community 1 - "Community 1"
Cohesion: 0.29
Nodes (4): distance, game, require, { TetrisGame }

### Community 2 - "Community 2"
Cohesion: 0.29
Nodes (6): code:sh (node game.test.mjs), Controls, Features, Run, Test, Tetris Smoke

## Knowledge Gaps
- **12 isolated node(s):** `TYPES`, `SHAPES`, `COLORS`, `POINTS`, `require` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TetrisGame` connect `Community 3` to `Community 0`, `Community 1`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.489) - this node is a cross-community bridge._
- **What connects `TYPES`, `SHAPES`, `COLORS` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._