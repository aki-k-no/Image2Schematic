const fs = require("fs");
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothGrid(values, width, length, passes) {
  let current = values.slice();
  for (let pass = 0; pass < passes; pass += 1) {
    const next = current.slice();
    for (let z = 0; z < length; z += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = current[z * width + x] * 4;
        let weight = 4;
        for (let dz = -1; dz <= 1; dz += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dz === 0) {
              continue;
            }
            const nx = x + dx;
            const nz = z + dz;
            if (nx < 0 || nx >= width || nz < 0 || nz >= length) {
              continue;
            }
            const w = Math.abs(dx) + Math.abs(dz) === 1 ? 2 : 1;
            sum += current[nz * width + nx] * w;
            weight += w;
          }
        }
        next[z * width + x] = sum / weight;
      }
    }
    current = next;
  }
  return current;
}

function asymmetricGaussian(x, z, cx, cz, rxFront, rxBack, rzLeft, rzRight, amplitude) {
  const dx = x - cx;
  const dz = z - cz;
  const rx = dx < 0 ? rxFront : rxBack;
  const rz = dz < 0 ? rzLeft : rzRight;
  const radial = (dx * dx) / (rx * rx) + (dz * dz) / (rz * rz);
  return amplitude * Math.exp(-radial);
}

function ridgeHeight(x, z, x0, x1, z0, z1, height, thickness) {
  const vx = x1 - x0;
  const vz = z1 - z0;
  const wx = x - x0;
  const wz = z - z0;
  const len2 = vx * vx + vz * vz;
  const t = clamp((wx * vx + wz * vz) / Math.max(len2, 1), 0, 1);
  const px = x0 + vx * t;
  const pz = z0 + vz * t;
  const dx = x - px;
  const dz = z - pz;
  const dist2 = dx * dx + dz * dz;
  const sigma2 = thickness * thickness;
  return height * Math.exp(-dist2 / (2 * sigma2));
}

function noise2d(x, z) {
  const s = Math.sin(x * 12.9898 + z * 78.233) * 43758.5453;
  return s - Math.floor(s);
}

function buildReferenceNoise(width, height, length, blocks) {
  const reference = new Float32Array(width * length);
  const counts = new Uint16Array(width * length);
  for (let x = 0; x < width; x += 1) {
    for (let z = 0; z < length; z += 1) {
      let highest = -1;
      for (let y = 0; y < height; y += 1) {
        const idx = indexOf(width, length, x, y, z);
        if (blocks[idx] !== 0 && y > highest) {
          highest = y;
        }
      }
      if (highest >= 0 && highest < height - 2) {
        const grid = z * width + x;
        reference[grid] = highest;
        counts[grid] = 1;
      }
    }
  }
  const smoothed = smoothGrid(reference, width, length, 2);
  const out = new Float32Array(width * length);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = counts[i] ? reference[i] - smoothed[i] : 0;
  }
  return out;
}

function main() {
  const inputPath = process.argv[2] || "./outputs/mountain1_debug.schematic";
  const outputPath = process.argv[3] || "./outputs/mountain1_debug_inferred3d.schematic";
  const { rootName, root } = readSchematic(inputPath);

  const width = root.Width.value;
  const height = root.Height.value;
  const length = root.Length.value;
  const sourceBlocks = root.Blocks.value;
  const referenceNoise = buildReferenceNoise(width, height, length, sourceBlocks);

  const blocks = Buffer.alloc(width * height * length, 0);
  const data = Buffer.alloc(width * height * length, 0);

  const heightMap = new Float32Array(width * length);
  const leftPeak = { x: 96, z: 63, rxFront: 108, rxBack: 82, rzLeft: 42, rzRight: 48 };
  const rightPeak = { x: 273, z: 69, rxFront: 92, rxBack: 78, rzLeft: 40, rzRight: 46 };

  for (let z = 0; z < length; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const grid = z * width + x;
      const foregroundRise =
        24 * Math.exp(-Math.pow((z - 22) / 18, 2)) *
        (0.76 + 0.24 * Math.cos((x - width * 0.52) / 48));
      const left =
        asymmetricGaussian(x, z, leftPeak.x, leftPeak.z, leftPeak.rxFront, leftPeak.rxBack, leftPeak.rzLeft, leftPeak.rzRight, 78) +
        asymmetricGaussian(x, z, leftPeak.x - 6, leftPeak.z - 4, 46, 34, 22, 24, 26);
      const right =
        asymmetricGaussian(x, z, rightPeak.x, rightPeak.z, rightPeak.rxFront, rightPeak.rxBack, rightPeak.rzLeft, rightPeak.rzRight, 64) +
        asymmetricGaussian(x, z, rightPeak.x + 3, rightPeak.z - 2, 38, 28, 19, 22, 18);
      const saddle =
        20 * Math.exp(-Math.pow((x - 188) / 54, 2) - Math.pow((z - 66) / 18, 2)) +
        ridgeHeight(x, z, 134, 238, 63, 69, 16, 17);
      const mountain = Math.max(left, right, saddle);
      const edgeFade = clamp(
        Math.min(x / 26, (width - 1 - x) / 26, z / 14, (length - 1 - z) / 11),
        0.18,
        1,
      );
      const largeNoise = (noise2d(x * 0.25, z * 0.25) - 0.5) * 9;
      const fineNoise = (noise2d(x * 0.8 + 31, z * 0.8 + 17) - 0.5) * 3;
      const inheritedNoise = referenceNoise[grid] * 0.18;
      const shaped =
        26 +
        foregroundRise +
        mountain * edgeFade +
        largeNoise +
        fineNoise +
        inheritedNoise;
      heightMap[grid] = clamp(shaped, 0, height - 1);
    }
  }

  const smoothHeight = smoothGrid(heightMap, width, length, 2);

  for (let z = 0; z < length; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const grid = z * width + x;
      const top = Math.round(smoothHeight[grid]);
      const dx = x < width - 1 ? smoothHeight[grid + 1] - smoothHeight[grid] : 0;
      const dz = z < length - 1 ? smoothHeight[grid + width] - smoothHeight[grid] : 0;
      const slope = Math.sqrt(dx * dx + dz * dz);

      for (let y = 0; y <= top; y += 1) {
        let blockId = 3;
        let blockData = 0;
        const depth = top - y;
        const highAlpine = top > 158;
        const forestBand = top >= 62 && top <= 108 && z <= 54;
        const shoulderForest = top >= 72 && top <= 128 && slope < 3.1 && noise2d(x * 0.37 + y, z * 0.37) > 0.62;

        if (depth === 0) {
          if (highAlpine && slope > 2.6) {
            blockId = 159;
            blockData = 12;
          } else if (highAlpine) {
            blockId = 2;
            blockData = 0;
          } else if (forestBand || shoulderForest) {
            blockId = 18;
            blockData = 3;
          } else if (slope > 3.4) {
            blockId = 3;
            blockData = 0;
          } else {
            blockId = 2;
            blockData = 0;
          }
        } else if (depth <= 2) {
          if (forestBand || shoulderForest) {
            blockId = noise2d(x + y * 0.5, z) > 0.84 ? 17 : 18;
            blockData = blockId === 17 ? 1 : 3;
          } else {
            blockId = 3;
            blockData = 0;
          }
        }

        const idx = indexOf(width, length, x, y, z);
        blocks[idx] = blockId;
        data[idx] = blockData;
      }
    }
  }

  root.Blocks.value = blocks;
  root.Data.value = data;
  if (root.AddBlocks) {
    root.AddBlocks.value = Buffer.alloc(Math.ceil(blocks.length / 2), 0);
  }

  writeSchematic(outputPath, rootName, root);

  let nonAir = 0;
  for (const id of blocks) {
    if (id !== 0) {
      nonAir += 1;
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
        nonAir,
      },
      null,
      2,
    ),
  );
}

main();
