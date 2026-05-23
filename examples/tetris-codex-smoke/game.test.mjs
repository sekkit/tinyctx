import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { TetrisGame } = require('./game.js');

function makeGame(queue = ['T', 'I', 'O', 'L', 'J', 'S', 'Z']) {
  return new TetrisGame({ queue, random: () => 0 });
}

function fillRow(game, row, except = []) {
  game.board[row] = Array.from({ length: game.cols }, (_, x) => except.includes(x) ? null : 'Z');
}

{
  const game = makeGame(['T', 'I']);
  assert.equal(game.current.type, 'T');
  assert.equal(game.current.col, 3);
  assert.equal(game.current.row, 0);
  assert.equal(game.peekNextType(), 'I');
}

{
  const game = makeGame(['O']);
  game.current.col = 0;
  assert.equal(game.move(-1), false);
  game.current.row = 18;
  assert.equal(game.softDrop(), false);
  assert.equal(game.board[18][0], 'O');
}

{
  const game = makeGame(['I']);
  game.current.col = 0;
  assert.equal(game.rotate(1), true);
  assert.deepEqual(game.current.matrix.map((row) => row.join('')), ['0010', '0010', '0010', '0010']);
  assert.ok(game.current.col >= 0);
}

{
  const game = makeGame(['I', 'O']);
  const distance = game.hardDrop();
  assert.equal(distance, 18);
  assert.equal(game.score, 36);
  assert.equal(game.board[19].filter(Boolean).length, 4);
  assert.equal(game.current.type, 'O');
}

{
  const game = makeGame(['I', 'O']);
  fillRow(game, 19, [3, 4, 5, 6]);
  game.current.row = 18;
  game.current.col = 3;
  game.lockPiece();
  assert.equal(game.lines, 1);
  assert.equal(game.score, 100);
  assert.ok(game.board[19].every((cell) => cell === null));
}

{
  const game = makeGame(['T', 'I', 'O']);
  assert.equal(game.hold(), true);
  assert.equal(game.holdType, 'T');
  assert.equal(game.current.type, 'I');
  assert.equal(game.hold(), false);
  game.hardDrop();
  assert.equal(game.hold(), true);
}

{
  const game = makeGame(['O']);
  game.board[0][4] = 'Z';
  game.spawnPiece('O');
  assert.equal(game.gameOver, true);
}

console.log('All tests passed');
