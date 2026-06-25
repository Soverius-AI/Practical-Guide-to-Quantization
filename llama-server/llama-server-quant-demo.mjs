#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { once } from "node:events";

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 18080;

const HELP = `
llama-server quantization demo runner

Starts llama-server for each GGUF model, runs the same prompts through the
OpenAI-compatible chat endpoint, and prints webinar-friendly timing numbers.

Usage:
  node scripts/llama-server-quant-demo.mjs \\
    --server /path/to/llama-server \\
    --q8 /models/model-q8_0.gguf \\
    --q4 /models/model-q4_k_m.gguf

Common options:
  --server PATH        llama-server binary. Default: LLAMA_SERVER or llama-server
  --q8 PATH           GGUF file for the Q8 run
  --q4 PATH           GGUF file for the Q4 run
  --model LABEL=PATH  Additional or custom model. Can be repeated
  --port N            Local port to use. Default: ${DEFAULT_PORT}
  --host HOST         Host to bind/connect. Default: ${DEFAULT_HOST}
  --ctx N             llama-server context size. Default: 4096
  --ngl N             GPU layers. Default: 999. Use 0 for CPU-only
  --threads N         Optional CPU thread count passed as -t
  --max-tokens N      Default max completion tokens. Default: 160
  --temperature N     Sampling temperature. Default: 0.2
  --prompt-set NAME   quick, prefill, or all. Default: quick
  --no-warmup         Skip the short warmup request
  --startup-timeout N Seconds to wait for llama-server. Default: 120

Pass extra llama-server args after "--":
  node scripts/llama-server-quant-demo.mjs --q4 model.gguf -- --flash-attn
`;

function parseArgs(argv) {
  const dashDash = argv.indexOf("--");
  const main = dashDash >= 0 ? argv.slice(0, dashDash) : argv;
  const passthrough = dashDash >= 0 ? argv.slice(dashDash + 1) : [];
  const options = {
    server: process.env.LLAMA_SERVER || "llama-server",
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    ctx: 4096,
    ngl: 999,
    threads: null,
    maxTokens: 160,
    temperature: 0.2,
    promptSet: "quick",
    warmup: true,
    startupTimeout: 120,
    models: [],
    passthrough,
  };

  for (let i = 0; i < main.length; i += 1) {
    const arg = main[i];
    const next = () => {
      if (i + 1 >= main.length) throw new Error(`Missing value for ${arg}`);
      i += 1;
      return main[i];
    };

    if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else if (arg === "--server") {
      options.server = next();
    } else if (arg === "--q8") {
      options.models.push({ label: "Q8", file: next() });
    } else if (arg === "--q4") {
      options.models.push({ label: "Q4", file: next() });
    } else if (arg === "--model") {
      const spec = next();
      const eq = spec.indexOf("=");
      if (eq <= 0) throw new Error("--model must be LABEL=/path/to/model.gguf");
      options.models.push({ label: spec.slice(0, eq), file: spec.slice(eq + 1) });
    } else if (arg === "--host") {
      options.host = next();
    } else if (arg === "--port") {
      options.port = Number(next());
    } else if (arg === "--ctx") {
      options.ctx = Number(next());
    } else if (arg === "--ngl") {
      options.ngl = Number(next());
    } else if (arg === "--threads") {
      options.threads = Number(next());
    } else if (arg === "--max-tokens") {
      options.maxTokens = Number(next());
    } else if (arg === "--temperature") {
      options.temperature = Number(next());
    } else if (arg === "--prompt-set") {
      options.promptSet = next();
    } else if (arg === "--no-warmup") {
      options.warmup = false;
    } else if (arg === "--startup-timeout") {
      options.startupTimeout = Number(next());
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function nowMs() {
  return Number(process.hrtime.bigint() / 1_000_000n);
}

function formatMs(ms) {
  if (!Number.isFinite(ms)) return "n/a";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatNumber(value, digits = 1) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

function estimateTokens(text) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return Math.max(1, Math.round(trimmed.length / 4));
}

async function fileInfo(file) {
  const stat = await fs.stat(file);
  return {
    bytes: stat.size,
    gib: stat.size / 1024 / 1024 / 1024,
  };
}

function promptSets(maxTokens) {
  const longContextBlock = `
Quantization notes:
- FP16 and BF16 store one weight in 2 bytes.
- Q8 usually stores one 8-bit code plus block scale metadata.
- Q4 stores a 4-bit code plus block scale metadata.
- Prefill reads the prompt and builds the Key/Value cache.
- Decode generates one token at a time and is often memory-bandwidth-bound.
- Good quantization protects sensitive tensors and compresses less sensitive tensors harder.
`.repeat(20);

  const prompts = {
    short: {
      name: "short-explain",
      maxTokens,
      user: "Explain why database indexes speed up reads but can slow down writes. Use one concrete example.",
    },
    prefill: {
      name: "long-prefill",
      maxTokens: Math.min(maxTokens, 140),
      user: `${longContextBlock}\n\nSummarize the notes in exactly three bullets for a senior engineer.`,
    },
    hard: {
      name: "hard-coding",
      maxTokens: Math.max(maxTokens, 220),
      user: "Write a JavaScript function parseDurationMs(input) that accepts strings like '1h 20m 5s', '750ms', and '2m'. Return milliseconds. Include three edge cases and explain them briefly.",
    },
  };

  return {
    quick: [prompts.short, prompts.prefill],
    prefill: [prompts.prefill],
    all: [prompts.short, prompts.prefill, prompts.hard],
  };
}

function serverArgs(options, modelFile) {
  const args = [
    "-m", modelFile,
    "--host", options.host,
    "--port", String(options.port),
    "-c", String(options.ctx),
    "-ngl", String(options.ngl),
  ];
  if (options.threads !== null) args.push("-t", String(options.threads));
  return args.concat(options.passthrough);
}

async function waitForServer(baseUrl, child, timeoutSeconds) {
  const deadline = Date.now() + timeoutSeconds * 1000;
  let lastError = null;

  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`llama-server exited early with code ${child.exitCode}`);
    }

    for (const route of ["/health", "/v1/models"]) {
      try {
        const res = await fetch(`${baseUrl}${route}`);
        if (res.ok) return;
      } catch (error) {
        lastError = error;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(`Timed out waiting for llama-server (${lastError?.message || "no response"})`);
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  child.kill("SIGTERM");
  const timeout = setTimeout(() => child.kill("SIGKILL"), 5000);
  try {
    await once(child, "exit");
  } finally {
    clearTimeout(timeout);
  }
}

async function chatCompletion(baseUrl, prompt, options) {
  const requestStart = nowMs();
  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "local",
      messages: [
        {
          role: "system",
          content: "You are a concise technical assistant. Answer directly.",
        },
        { role: "user", content: prompt.user },
      ],
      temperature: options.temperature,
      max_tokens: prompt.maxTokens,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`chat completion failed: HTTP ${response.status} ${body.slice(0, 500)}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let output = "";
  let firstTokenAt = null;
  let usageTokens = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const rawLine of event.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        let parsed;
        try {
          parsed = JSON.parse(data);
        } catch {
          continue;
        }
        if (parsed.usage?.completion_tokens) {
          usageTokens = parsed.usage.completion_tokens;
        }
        const delta = parsed.choices?.[0]?.delta?.content ?? parsed.choices?.[0]?.message?.content ?? "";
        if (delta) {
          if (firstTokenAt === null) firstTokenAt = nowMs();
          output += delta;
        }
      }
    }
  }

  const end = nowMs();
  const completionTokens = usageTokens ?? estimateTokens(output);
  const generationMs = firstTokenAt === null ? end - requestStart : end - firstTokenAt;

  return {
    prompt: prompt.name,
    ttftMs: firstTokenAt === null ? null : firstTokenAt - requestStart,
    totalMs: end - requestStart,
    generationMs,
    completionTokens,
    tokensAreEstimated: usageTokens === null,
    tokensPerSecond: generationMs > 0 ? completionTokens / (generationMs / 1000) : null,
    preview: output.replace(/\s+/g, " ").trim().slice(0, 180),
  };
}

async function runModel(model, options, prompts, outputDir) {
  const baseUrl = `http://${options.host}:${options.port}`;
  const info = await fileInfo(model.file);
  const args = serverArgs(options, model.file);
  const logFile = path.join(outputDir, `llama-server-${model.label.replace(/[^a-z0-9_-]+/gi, "_")}.log`);
  const logHandle = await fs.open(logFile, "w");

  console.log(`\n=== ${model.label} ===`);
  console.log(`file: ${model.file}`);
  console.log(`size: ${info.gib.toFixed(2)} GiB`);
  console.log(`start: ${options.server} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);

  const start = nowMs();
  const child = spawn(options.server, args, { stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", (chunk) => logHandle.write(chunk));
  child.stderr.on("data", (chunk) => logHandle.write(chunk));

  try {
    await waitForServer(baseUrl, child, options.startupTimeout);
    const startupMs = nowMs() - start;
    console.log(`ready after ${formatMs(startupMs)}`);

    if (options.warmup) {
      await chatCompletion(baseUrl, { name: "warmup", user: "Reply with READY.", maxTokens: 8 }, options);
      console.log("warmup complete");
    }

    const results = [];
    for (const prompt of prompts) {
      process.stdout.write(`run ${prompt.name} ... `);
      const result = await chatCompletion(baseUrl, prompt, options);
      results.push(result);
      console.log(
        `TTFT ${formatMs(result.ttftMs)}, total ${formatMs(result.totalMs)}, ` +
        `${result.completionTokens}${result.tokensAreEstimated ? "~" : ""} tokens, ` +
        `${formatNumber(result.tokensPerSecond)} tok/s`,
      );
    }

    return {
      label: model.label,
      file: model.file,
      sizeGiB: info.gib,
      startupMs,
      logFile,
      results,
    };
  } finally {
    await stopServer(child);
    await logHandle.close();
  }
}

function printSummary(allResults) {
  console.log("\nSummary");
  console.log("| model | size | prompt | TTFT | total | out tokens | tok/s |");
  console.log("|---|---:|---|---:|---:|---:|---:|");
  for (const model of allResults) {
    for (const result of model.results) {
      console.log(
        `| ${model.label} | ${model.sizeGiB.toFixed(2)} GiB | ${result.prompt} | ` +
        `${formatMs(result.ttftMs)} | ${formatMs(result.totalMs)} | ` +
        `${result.completionTokens}${result.tokensAreEstimated ? "~" : ""} | ` +
        `${formatNumber(result.tokensPerSecond)} |`,
      );
    }
  }

  console.log("\nNotes:");
  console.log("- A ~ after output tokens means the script estimated token count from text length because usage was not returned.");
  console.log("- For VRAM on NVIDIA, run this in another terminal during the demo: watch -n 0.5 nvidia-smi");
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(HELP.trim());
    return;
  }
  if (options.models.length === 0) {
    throw new Error("Provide at least one model with --q4, --q8, or --model LABEL=PATH");
  }
  if (!Number.isFinite(options.port) || options.port <= 0) throw new Error("--port must be a number");
  if (!Number.isFinite(options.ctx) || options.ctx <= 0) throw new Error("--ctx must be a number");

  const sets = promptSets(options.maxTokens);
  const prompts = sets[options.promptSet];
  if (!prompts) throw new Error(`Unknown --prompt-set ${options.promptSet}; use quick, prefill, or all`);

  const outputDir = path.resolve("outputs", "llama-server-demo");
  await fs.mkdir(outputDir, { recursive: true });

  const allResults = [];
  for (const model of options.models) {
    allResults.push(await runModel(model, options, prompts, outputDir));
  }

  printSummary(allResults);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const resultFile = path.join(outputDir, `results-${stamp}.json`);
  await fs.writeFile(resultFile, JSON.stringify({ options, results: allResults }, null, 2));
  console.log(`\nSaved JSON results: ${resultFile}`);
}

main().catch((error) => {
  console.error(`\nERROR: ${error.message}`);
  process.exitCode = 1;
});
