(function (global) {
  'use strict';

  const COLS = 10;
  const ROWS = 20;
  const TYPES = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
  const SHAPES = Object.freeze({
    I: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    J: [[1,0,0],[1,1,1],[0,0,0]],
    L: [[0,0,1],[1,1,1],[0,0,0]],
    O: [[1,1],[1,1]],
    S: [[0,1,1],[1,1,0],[0,0,0]],
    T: [[0,1,0],[1,1,1],[0,0,0]],
    Z: [[1,1,0],[0,1,1],[0,0,0]],
  });
  const COLORS = Object.freeze({ I: '#22d3ee', J: '#60a5fa', L: '#fb923c', O: '#facc15', S: '#4ade80', T: '#c084fc', Z: '#f87171' });
  const POINTS = [0, 100, 300, 500, 800];

  function cloneMatrix(matrix) {
    return matrix.map((row) => row.slice());
  }

  function createBoard(rows = ROWS, cols = COLS) {
    return Array.from({ length: rows }, () => Array(cols).fill(null));
  }

  function rotateMatrix(matrix, direction = 1) {
    const size = matrix.length;
    const rotated = Array.from({ length: size }, () => Array(size).fill(0));
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        if (direction > 0) rotated[x][size - 1 - y] = matrix[y][x];
        else rotated[size - 1 - x][y] = matrix[y][x];
      }
    }
    return rotated;
  }

  class TetrisGame {
    constructor(options = {}) {
      this.cols = options.cols || COLS;
      this.rows = options.rows || ROWS;
      this.random = options.random || Math.random;
      this.initialQueue = Array.isArray(options.queue) ? options.queue.slice() : [];
      this.restart();
    }

    restart() {
      this.board = createBoard(this.rows, this.cols);
      this.queue = this.initialQueue.slice();
      this.score = 0;
      this.lines = 0;
      this.level = 1;
      this.dropInterval = 800;
      this.holdType = null;
      this.holdLocked = false;
      this.paused = false;
      this.gameOver = false;
      this.current = null;
      this.spawnPiece();
    }

    makeBag() {
      const bag = TYPES.slice();
      for (let i = bag.length - 1; i > 0; i -= 1) {
        const j = Math.floor(this.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
      return bag;
    }

    ensureQueue() {
      while (this.queue.length < 7) this.queue.push(...this.makeBag());
    }

    nextType() {
      this.ensureQueue();
      return this.queue.shift();
    }

    peekNextType() {
      this.ensureQueue();
      return this.queue[0];
    }

    spawnPiece(type = this.nextType()) {
      const matrix = cloneMatrix(SHAPES[type]);
      this.current = { type, matrix, row: 0, col: Math.floor((this.cols - matrix[0].length) / 2) };
      this.holdLocked = false;
      if (this.collides(this.current, 0, 0, matrix)) this.gameOver = true;
      return !this.gameOver;
    }

    cells(piece = this.current, rowOffset = 0, colOffset = 0, matrix = piece && piece.matrix) {
      if (!piece) return [];
      const blocks = [];
      matrix.forEach((row, y) => row.forEach((value, x) => {
        if (value) blocks.push({ x: piece.col + colOffset + x, y: piece.row + rowOffset + y });
      }));
      return blocks;
    }

    collides(piece = this.current, rowOffset = 0, colOffset = 0, matrix = piece && piece.matrix) {
      return this.cells(piece, rowOffset, colOffset, matrix).some(({ x, y }) => (
        x < 0 || x >= this.cols || y >= this.rows || (y >= 0 && this.board[y][x])
      ));
    }

    move(direction) {
      if (this.paused || this.gameOver || this.collides(this.current, 0, direction)) return false;
      this.current.col += direction;
      return true;
    }

    softDrop() {
      if (this.paused || this.gameOver) return false;
      if (!this.collides(this.current, 1, 0)) {
        this.current.row += 1;
        return true;
      }
      this.lockPiece();
      return false;
    }

    hardDrop() {
      if (this.paused || this.gameOver) return 0;
      let distance = 0;
      while (!this.collides(this.current, 1, 0)) {
        this.current.row += 1;
        distance += 1;
      }
      this.score += distance * 2;
      this.lockPiece();
      return distance;
    }

    rotate(direction = 1) {
      if (this.paused || this.gameOver || this.current.type === 'O') return false;
      const rotated = rotateMatrix(this.current.matrix, direction);
      const kicks = [0, -1, 1, -2, 2];
      for (const kick of kicks) {
        if (!this.collides(this.current, 0, kick, rotated)) {
          this.current.matrix = rotated;
          this.current.col += kick;
          return true;
        }
      }
      return false;
    }

    lockPiece() {
      for (const { x, y } of this.cells()) {
        if (y < 0) {
          this.gameOver = true;
          return;
        }
        this.board[y][x] = this.current.type;
      }
      const cleared = this.clearLines();
      this.applyScore(cleared);
      this.spawnPiece();
    }

    clearLines() {
      const kept = this.board.filter((row) => row.some((cell) => !cell));
      const cleared = this.rows - kept.length;
      while (kept.length < this.rows) kept.unshift(Array(this.cols).fill(null));
      this.board = kept;
      return cleared;
    }

    applyScore(cleared) {
      if (cleared > 0) {
        this.lines += cleared;
        this.score += POINTS[cleared] * this.level;
        this.level = Math.floor(this.lines / 10) + 1;
        this.dropInterval = Math.max(100, 800 - (this.level - 1) * 70);
      }
    }

    hold() {
      if (this.paused || this.gameOver || this.holdLocked) return false;
      const outgoing = this.current.type;
      if (!this.holdType) {
        this.holdType = outgoing;
        this.spawnPiece();
      } else {
        const incoming = this.holdType;
        this.holdType = outgoing;
        this.spawnPiece(incoming);
      }
      this.holdLocked = true;
      return true;
    }

    togglePause() {
      if (!this.gameOver) this.paused = !this.paused;
      return this.paused;
    }

    getCurrentCells() {
      return this.cells();
    }

    getGhostCells() {
      if (!this.current) return [];
      let rowOffset = 0;
      while (!this.collides(this.current, rowOffset + 1, 0)) rowOffset += 1;
      return this.cells(this.current, rowOffset, 0);
    }
  }

  TetrisGame.COLS = COLS;
  TetrisGame.ROWS = ROWS;
  TetrisGame.TYPES = TYPES;
  TetrisGame.SHAPES = SHAPES;
  TetrisGame.COLORS = COLORS;
  TetrisGame.createBoard = createBoard;
  TetrisGame.rotateMatrix = rotateMatrix;

  if (typeof module !== 'undefined' && module.exports) module.exports = { TetrisGame, createBoard, rotateMatrix, SHAPES, TYPES };
  global.TetrisGame = TetrisGame;
})(typeof globalThis !== 'undefined' ? globalThis : window);
