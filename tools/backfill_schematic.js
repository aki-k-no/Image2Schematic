const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

function readSchematic(filePath) {
  const buf = zlib.gunzipSync(fs.readFileSync(filePath));
  let offset = 0;

  const readU8 = () => buf[offset++];
  const readI8 = () => {
    const value = buf.readInt8(offset);
    offset += 1;
    return value;
  };
  const readI16 = () => {
    const value = buf.readInt16BE(offset);
    offset += 2;
    return value;
  };
  const readI32 = () => {
    const value = buf.readInt32BE(offset);
    offset += 4;
    return value;
  };
  const readString = () => {
    const length = readI16();
    const value = buf.toString("utf8", offset, offset + length);
    offset += length;
    return value;
  };

  function readPayload(tagType) {
    switch (tagType) {
      case 1:
        return readI8();
      case 2:
        return readI16();
      case 3:
        return readI32();
      case 7: {
        const length = readI32();
        const value = Buffer.from(buf.subarray(offset, offset + length));
        offset += length;
        return value;
      }
      case 8:
        return readString();
      case 9: {
        const elementType = readU8();
        const length = readI32();
        const values = [];
        for (let i = 0; i < length; i += 1) {
          values.push(readPayload(elementType));
        }
        return { type: 9, elementType, values };
      }
      case 10: {
        const payload = {};
        while (true) {
          const childType = readU8();
          if (childType === 0) {
            break;
          }
          const name = readString();
          payload[name] = { type: childType, value: readPayload(childType) };
        }
        return payload;
      }
      case 11: {
        const length = readI32();
        const values = [];
        for (let i = 0; i < length; i += 1) {
          values.push(readI32());
        }
        return values;
      }
      default:
        throw new Error(`Unsupported NBT tag ${tagType}`);
    }
  }

  const rootType = readU8();
  if (rootType !== 10) {
    throw new Error(`Unexpected root type: ${rootType}`);
  }
  const rootName = readString();
  const root = readPayload(rootType);
  return { rootName, root };
}

function writeSchematic(filePath, rootName, root) {
  const chunks = [];

  const pushU8 = (value) => chunks.push(Buffer.from([value & 0xff]));
  const pushI8 = (value) => {
    const buf = Buffer.alloc(1);
    buf.writeInt8(value, 0);
    chunks.push(buf);
  };
  const pushI16 = (value) => {
    const buf = Buffer.alloc(2);
    buf.writeInt16BE(value, 0);
    chunks.push(buf);
  };
  const pushI32 = (value) => {
    const buf = Buffer.alloc(4);
    buf.writeInt32BE(value, 0);
    chunks.push(buf);
  };
  const pushString = (value) => {
    const buf = Buffer.from(value, "utf8");
    pushI16(buf.length);
    chunks.push(buf);
  };

  function writePayload(tagType, value) {
    switch (tagType) {
      case 1:
        pushI8(value);
        break;
      case 2:
        pushI16(value);
        break;
      case 3:
        pushI32(value);
        break;
      case 7:
        pushI32(value.length);
        chunks.push(Buffer.from(value));
        break;
      case 8:
        pushString(value);
        break;
      case 9:
        pushU8(value.elementType);
        pushI32(value.values.length);
        for (const item of value.values) {
          writePayload(value.elementType, item);
        }
        break;
      case 10:
        for (const [name, entry] of Object.entries(value)) {
          pushU8(entry.type);
          pushString(name);
          writePayload(entry.type, entry.value);
        }
        pushU8(0);
        break;
      case 11:
        pushI32(value.length);
        for (const item of value) {
          pushI32(item);
        }
        break;
      default:
        throw new Error(`Unsupported tag for write: ${tagType}`);
    }
  }

  pushU8(10);
  pushString(rootName);
  writePayload(10, root);
  fs.writeFileSync(filePath, zlib.gzipSync(Buffer.concat(chunks)));
}

function indexOf(width, length, x, y, z) {
  return x + z * width + y * width * length;
}

function smoothSeries(values, passes) {
  let current = values.slice();
  for (let pass = 0; pass < passes; pass += 1) {
    const next = current.slice();
    for (let i = 0; i < current.length; i += 1) {
      let weight = 0;
      let sum = 0;
      for (let d = -2; d <= 2; d += 1) {
        const j = i + d;
        if (j < 0 || j >= current.length || current[j] < 0) {
          continue;
        }
        const w = d === 0 ? 4 : Math.abs(d) === 1 ? 2 : 1;
        sum += current[j] * w;
        weight += w;
      }
      if (weight > 0) {
        next[i] = sum / weight;
      }
    }
    current = next;
  }
  return current;
}

function smoothGrid(grid, mask, width, height, passes) {
  let current = grid.slice();
  for (let pass = 0; pass < passes; pass += 1) {
    const next = current.slice();
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if (!mask[idx]) {
          continue;
        }
        let sum = current[idx] * 4;
        let weight = 4;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) {
              continue;
            }
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
              continue;
            }
            const nIdx = ny * width + nx;
            if (!mask[nIdx]) {
              continue;
            }
            const w = Math.abs(dx) + Math.abs(dy) === 1 ? 2 : 1;
            sum += current[nIdx] * w;
            weight += w;
          }
        }
        next[idx] = sum / weight;
      }
    }
    current = next;
  }
  return current;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function majorityBlock(values) {
  const counts = new Map();
  let winner = values[0];
  let best = 0;
  for (const value of values) {
    const next = (counts.get(value) || 0) + 1;
    counts.set(value, next);
    if (next > best) {
      best = next;
      winner = value;
    }
  }
  return winner;
}

function pickFillBlock(patternIds, patternData, surfaceIndex, depthFromSurface) {
  const surfaceId = patternIds[surfaceIndex];
  const surfaceData = patternData[surfaceIndex];
  if (depthFromSurface <= 1) {
    return [surfaceId, surfaceData];
  }
  if (surfaceId === 18 || surfaceId === 17 || surfaceId === 161) {
    return depthFromSurface <= 3 ? [surfaceId, surfaceData] : [3, 0];
  }
  if (surfaceId === 2) {
    return depthFromSurface <= 2 ? [2, 0] : [3, 0];
  }
  if (surfaceId === 159 || surfaceId === 1 || surfaceId === 4) {
    return [surfaceId, surfaceData];
  }
  return [3, 0];
}

function drawLine(blocks, data, width, length, start, end, patternIds, patternData, surfaceIndex) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const dz = end[2] - start[2];
  const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz), 1);
  let added = 0;
  for (let step = 1; step <= steps; step += 1) {
    const t = step / steps;
    const x = Math.round(start[0] + dx * t);
    const y = Math.round(start[1] + dy * t);
    const z = Math.round(start[2] + dz * t);
    if (x < 0 || y < 0 || z < 0) {
      continue;
    }
    const index = indexOf(width, length, x, y, z);
    if (index < 0 || index >= blocks.length) {
      continue;
    }
    if (blocks[index] !== 0) {
      continue;
    }
    const depthFromSurface = Math.max(1, Math.round(step));
    const [fillId, fillData] = pickFillBlock(patternIds, patternData, surfaceIndex, depthFromSurface);
    blocks[index] = fillId;
    data[index] = fillData;
    added += 1;
  }
  return added;
}

function fillColumnGaps(blocks, data, width, height, length, maxGap) {
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      let first = -1;
      let lastFilled = -1;
      for (let z = 0; z < length; z += 1) {
        const idx = indexOf(width, length, x, y, z);
        if (blocks[idx] === 0) {
          continue;
        }
        if (first < 0) {
          first = z;
        }
        if (lastFilled >= 0 && z - lastFilled - 1 > 0 && z - lastFilled - 1 <= maxGap) {
          const fillId = blocks[indexOf(width, length, x, y, lastFilled)];
          const fillData = data[indexOf(width, length, x, y, lastFilled)];
          for (let gap = lastFilled + 1; gap < z; gap += 1) {
            const gapIndex = indexOf(width, length, x, y, gap);
            blocks[gapIndex] = fillId;
            data[gapIndex] = fillData;
          }
        }
        lastFilled = z;
      }
      if (first < 0) {
        continue;
      }
    }
  }
}

function fillEnclosedHoles(blocks, data, width, height, length, iterations, minNeighbors) {
  const neighbors = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const updates = [];
    for (let x = 1; x < width - 1; x += 1) {
      for (let y = 1; y < height - 1; y += 1) {
        for (let z = 1; z < length - 1; z += 1) {
          const idx = indexOf(width, length, x, y, z);
          if (blocks[idx] !== 0) {
            continue;
          }
          const blockVotes = [];
          for (const [dx, dy, dz] of neighbors) {
            const nIndex = indexOf(width, length, x + dx, y + dy, z + dz);
            if (blocks[nIndex] !== 0) {
              blockVotes.push(`${blocks[nIndex]}:${data[nIndex]}`);
            }
          }
          if (blockVotes.length < minNeighbors) {
            continue;
          }
          const pick = majorityBlock(blockVotes);
          const [fillId, fillData] = pick.split(":").map(Number);
          updates.push([idx, fillId, fillData]);
        }
      }
    }
    if (!updates.length) {
      break;
    }
    for (const [idx, fillId, fillData] of updates) {
      blocks[idx] = fillId;
      data[idx] = fillData;
    }
  }
}

function main() {
  const inputPath = process.argv[2] || path.join("outputs", "mountain1_debug.schematic");
  const outputPath = process.argv[3] || path.join("outputs", "mountain1_debug_surface_preserved.schematic");
  const { rootName, root } = readSchematic(inputPath);
  const width = root.Width.value;
  const height = root.Height.value;
  const length = root.Length.value;
  const blocks = Buffer.from(root.Blocks.value);
  const data = Buffer.from(root.Data.value);
  const addBlocks = root.AddBlocks ? Buffer.from(root.AddBlocks.value) : null;

  const minYAtX = Array(width).fill(-1);
  const maxYAtX = Array(width).fill(-1);
  const occupancy = new Uint8Array(width * height);
  const firstZ = new Int16Array(width * height);
  const lastZ = new Int16Array(width * height);
  const surfaceZ = new Float32Array(width * height);
  const crestYAtX = Array(width).fill(-1);
  const baseYAtX = Array(width).fill(-1);

  for (let i = 0; i < firstZ.length; i += 1) {
    firstZ[i] = -1;
    lastZ[i] = -1;
    surfaceZ[i] = -1;
  }

  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const gridIndex = y * width + x;
      for (let z = 0; z < length; z += 1) {
        const index = indexOf(width, length, x, y, z);
        if (blocks[index] === 0) {
          continue;
        }
        occupancy[gridIndex] = 1;
        if (firstZ[gridIndex] < 0) {
          firstZ[gridIndex] = z;
        }
        lastZ[gridIndex] = z;
        surfaceZ[gridIndex] = firstZ[gridIndex];
        if (minYAtX[x] < 0 || y < minYAtX[x]) {
          minYAtX[x] = y;
        }
        if (y > maxYAtX[x]) {
          maxYAtX[x] = y;
        }
      }
    }
  }

  const smoothMinY = smoothSeries(minYAtX, 3);
  const smoothMaxY = smoothSeries(maxYAtX, 2);
  for (let x = 0; x < width; x += 1) {
    crestYAtX[x] = smoothMinY[x];
    baseYAtX[x] = smoothMaxY[x];
  }

  let extrudedRows = 0;
  let addedBlocks = 0;
  for (let x = 0; x < width; x += 1) {
    if (crestYAtX[x] < 0 || baseYAtX[x] < 0) {
      continue;
    }
    for (let y = 0; y < height; y += 1) {
      const gridIndex = y * width + x;
      if (!occupancy[gridIndex]) {
        continue;
      }
      const first = firstZ[gridIndex];
      const last = lastZ[gridIndex];
      const crestY = crestYAtX[x];
      const baseY = baseYAtX[x];
      const relativeDown = clamp((y - crestY) / Math.max(baseY - crestY, 1), 0, 1);
      const prominence = clamp((92 - crestY) / 92, 0, 1);
      const edgeFactor = clamp(Math.min(x / 30, (width - 1 - x) / 30), 0.25, 1.0);
      const currentSpan = last - first + 1;
      const baseThickness = 8 + prominence * 14;
      const slopeThickness = 14 + prominence * 34;
      const targetThickness =
        baseThickness +
        slopeThickness * Math.pow(relativeDown, 1.35) * edgeFactor +
        7 * Math.exp(-Math.pow(relativeDown - 0.58, 2) / 0.05);

      const left = x > 0 && surfaceZ[gridIndex - 1] >= 0 ? surfaceZ[gridIndex - 1] : surfaceZ[gridIndex];
      const right = x + 1 < width && surfaceZ[gridIndex + 1] >= 0 ? surfaceZ[gridIndex + 1] : surfaceZ[gridIndex];
      const up = y > 0 && surfaceZ[(y - 1) * width + x] >= 0 ? surfaceZ[(y - 1) * width + x] : surfaceZ[gridIndex];
      const down = y + 1 < height && surfaceZ[(y + 1) * width + x] >= 0 ? surfaceZ[(y + 1) * width + x] : surfaceZ[gridIndex];
      const dzdx = (right - left) * 0.5;
      const dzdy = (down - up) * 0.5;
      const curvatureBoost = clamp((Math.abs(dzdx) + Math.abs(dzdy)) * 0.35, 0, 10);
      const extension = clamp(
        Math.round(Math.max(currentSpan, targetThickness + curvatureBoost)),
        currentSpan,
        length - 1 - first,
      );
      const start = [x, y, last];
      const end = [
        x,
        y,
        clamp(Math.round(first + extension), last, length - 1),
      ];

      const patternIds = [];
      const patternData = [];
      for (let z = first; z <= last; z += 1) {
        const sourceIndex = indexOf(width, length, x, y, z);
        patternIds.push(blocks[sourceIndex]);
        patternData.push(data[sourceIndex]);
      }
      addedBlocks += drawLine(blocks, data, width, length, start, end, patternIds, patternData, 0);
      extrudedRows += 1;
    }
  }

  fillColumnGaps(blocks, data, width, height, length, 6);
  fillEnclosedHoles(blocks, data, width, height, length, 2, 4);

  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const gridIndex = y * width + x;
      const frontLimit = firstZ[gridIndex];
      if (!occupancy[gridIndex]) {
        for (let z = 0; z < length; z += 1) {
          const idx = indexOf(width, length, x, y, z);
          blocks[idx] = 0;
          data[idx] = 0;
        }
        continue;
      }
      if (frontLimit < 0) {
        continue;
      }
      for (let z = 0; z < frontLimit; z += 1) {
        const idx = indexOf(width, length, x, y, z);
        blocks[idx] = 0;
        data[idx] = 0;
      }
    }
  }

  root.Blocks.value = blocks;
  root.Data.value = data;
  if (addBlocks) {
    root.AddBlocks.value = addBlocks;
  }

  writeSchematic(outputPath, rootName, root);
  let finalNonAir = 0;
  for (const id of blocks) {
    if (id !== 0) {
      finalNonAir += 1;
    }
  }

  console.log(
    JSON.stringify(
      {
        inputPath,
        outputPath,
        width,
        height,
        length,
        extrudedRows,
        addedBlocks,
        finalNonAir,
      },
      null,
      2,
    ),
  );
}

main();
