// Sokoban solver: push-optimal A* per room. A level is a hallway with small rooms
// off it; a box that leaves its room never comes back, so each room is independent
// and the level optimum is the sum of its rooms'. Deterministic: budgeted by states
// expanded, never by the wall clock.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
const ROOM_BUDGET = 400_000;
const GREEDY_BUDGET = 200_000;
interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}
function liveSquares(L: Level): Uint8Array {
  const { w, wall, goals } = L;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}
function goalDistances(L: Level): Int32Array[] {
  const { w, wall, goals } = L;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p] + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}
function estimate(boxes: number[], dist: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n <= 5) {
    const order = boxes.map((_, i) => i);
    let best = Infinity;
    const permute = (k: number): void => {
      if (k === n) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[i]![boxes[order[i]!]!]!;
          if (d < 0) return;
          s += d;
        }
        if (s < best) best = s;
        return;
      }
      for (let i = k; i < n; i++) {
        [order[k], order[i]] = [order[i]!, order[k]!];
        permute(k + 1);
        [order[k], order[i]] = [order[i]!, order[k]!];
      }
    };
    permute(0);
    return best === Infinity ? 1e9 : best;
  }
  let s = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (const d of dist) {
      const v = d[b]!;
      if (v >= 0 && v < m) m = v;
    }
    if (m === Infinity) return 1e9;
    s += m;
  }
  return s;
}
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p]! + s;
  return s;
}
function searchPushes(
  L: Level,
  maxExpansions: number,
  greedy: boolean,
): { pushes: number; moves: string } | null {
  const { w, wall, goals } = L;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(L);
  if (L.boxes.some((b) => !live[b])) return null;
  const dist = goalDistances(L);
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  const occupied = new Uint8Array(size);
  const makeRegion = () => {
    const seen = new Int32Array(size);
    const stack = new Int32Array(size);
    let stamp = 0;
    const low = (from: number): number => {
      stamp++;
      let lowSq = from;
      let n = 0;
      stack[n++] = from;
      seen[from] = stamp;
      while (n > 0) {
        const p = stack[--n];
        if (p < lowSq) lowSq = p;
        if (p >= w && !wall[p - w] && !occupied[p - w] && seen[p - w] !== stamp) {
          seen[p - w] = stamp;
          stack[n++] = p - w;
        }
        if (p + w < size && !wall[p + w] && !occupied[p + w] && seen[p + w] !== stamp) {
          seen[p + w] = stamp;
          stack[n++] = p + w;
        }
        if (p % w > 0 && !wall[p - 1] && !occupied[p - 1] && seen[p - 1] !== stamp) {
          seen[p - 1] = stamp;
          stack[n++] = p - 1;
        }
        if (p % w < w - 1 && !wall[p + 1] && !occupied[p + 1] && seen[p + 1] !== stamp) {
          seen[p + 1] = stamp;
          stack[n++] = p + 1;
        }
      }
      return lowSq;
    };
    const has = (sq: number): boolean => seen[sq] === stamp;
    return { low, has };
  };
  const reachNow = makeRegion();
  const reachNext = makeRegion();
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const sBoxes: number[][] = [L.boxes.slice()];
  const sPlayer: number[] = [L.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const best = new Map<number, number>();
  const closed = new Set<number>();
  let seq = 0;
  const h0 = estimate(L.boxes, dist);
  const rankOf = (g: number, h: number): number => (greedy ? h : g + h);
  for (const b of L.boxes) occupied[b] = 1;
  const low0 = reachNow.low(L.player);
  for (const b of L.boxes) occupied[b] = 0;
  const k0 = keyOf(L.boxes, low0);
  best.set(k0, 0);
  heapPush([rankOf(0, h0), 0, seq++, 0]);
  if (solved(L.boxes)) return { pushes: 0, moves: '' };
  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < maxExpansions) {
    const [, , , s] = heapPop();
    const bs = sBoxes[s]!;
    const g = sG[s]!;
    for (const b of bs) occupied[b] = 1;
    const low = reachNow.low(sPlayer[s]!);
    const k = keyOf(bs, low);
    if (closed.has(k) || best.get(k) !== g) {
      for (const x of bs) occupied[x] = 0;
      continue;
    }
    closed.add(k);
    expanded++;
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (!reachNow.has(b - d) || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const dead = frozenBlock(w, wall, occupied, goal, t);
        let next: number[] | null = null;
        let nk = -1;
        if (!dead) {
          next = bs.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          nk = keyOf(next, reachNext.low(b));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (dead || nk < 0 || next === null) continue;
        if (closed.has(nk)) continue;
        const ng = g + 1;
        const prev = best.get(nk);
        if (prev !== undefined && prev <= ng) continue;
        best.set(nk, ng);
        const idx = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(ng);
        if (solved(next)) {
          found = idx;
          break;
        }
        heapPush([rankOf(ng, estimate(next, dist)), -ng, seq++, idx]);
      }
    }
    for (const x of bs) occupied[x] = 0;
  }
  if (found < 0) {
    return null;
  }
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = L.player;
  let boxes = L.boxes.slice();
  for (const s of path) {
    const from = sFrom[s]!;
    const to = sTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[2].toUpperCase();
    player = from;
    boxes = sBoxes[s]!.slice();
  }
  return { pushes: path.length, moves };
}
interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}
function roomsOf(L: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = L;
  const live = liveSquares(L);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}
function solveByRooms(L: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(L);
  if (rooms === null) return null;
  let player = L.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: L.w,
      h: L.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = searchPushes(sub, ROOM_BUDGET, false);
    if (solved === null) solved = searchPushes(sub, GREEDY_BUDGET, true);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    let p = player;
    for (const ch of solved.moves.toLowerCase()) p += ch === 'u' ? -L.w : ch === 'd' ? L.w : ch === 'l' ? -1 : 1;
    player = p;
    room.done = true;
  }
  return { pushes, moves };
}
function solveLevel(text: string): string {
  const L = parseLevel(text);
  if (L.boxes.every((b) => L.goals.includes(b))) return '';
  const byRooms = solveByRooms(L);
  if (byRooms !== null) return byRooms.moves;
  const whole = searchPushes(L, ROOM_BUDGET, false) ?? searchPushes(L, GREEDY_BUDGET, true);
  return whole === null ? '' : whole.moves;
}
for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
