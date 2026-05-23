# Tetris Smoke

A dependency-free browser Tetris implementation in plain HTML, CSS, and JavaScript.

## Run

Open `index.html` in a browser. No package install or network access is required.

## Controls

- `ArrowLeft` / `ArrowRight`: move
- `ArrowDown`: soft drop
- `ArrowUp` / `X`: rotate clockwise
- `Z`: rotate counter-clockwise
- `Space`: hard drop
- `C`: hold piece
- `P`: pause or resume
- `R`: restart

## Features

- 10x20 board and all seven tetrominoes
- Rotation with simple wall kicks
- Collision detection, hard drop, line clearing, scoring, and level speed progression
- Pause/resume, restart, next preview, hold lockout, and game over state

## Test

```sh
node game.test.mjs
```
